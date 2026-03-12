import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips

from cleanfid import fid as clean_fid


class PairedImageDataset(Dataset):
    def __init__(self, ref_dir, gen_dir, transform=None):
        self.ref_dir = ref_dir
        self.gen_dir = gen_dir
        self.transform = transform
        
        self.ref_images = sorted([f for f in os.listdir(ref_dir) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.gen_images = sorted([f for f in os.listdir(gen_dir) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        self.num_ref = len(self.ref_images)
        self.num_gen = len(self.gen_images)
        print(f"Found {self.num_ref} reference images, {self.num_gen} generated images")
        
    def __len__(self):
        return self.num_gen
    
    def __getitem__(self, idx):
        ref_idx = idx % self.num_ref
        
        ref_path = os.path.join(self.ref_dir, self.ref_images[ref_idx])
        gen_path = os.path.join(self.gen_dir, self.gen_images[idx])
        
        ref_img = Image.open(ref_path).convert('RGB')
        gen_img = Image.open(gen_path).convert('RGB')
        
        if ref_img.size != gen_img.size:
            ref_img = ref_img.resize(gen_img.size, Image.LANCZOS)
        
        if self.transform:
            ref_tensor = self.transform(ref_img)
            gen_tensor = self.transform(gen_img)
        else:
            ref_tensor = transforms.ToTensor()(ref_img)
            gen_tensor = transforms.ToTensor()(gen_img)
            
        return {
            'ref': ref_tensor,
            'gen': gen_tensor,
            'ref_np': np.array(ref_img),
            'gen_np': np.array(gen_img),
        }


def compute_pairwise_metrics(ref_dir, gen_dir, device='cuda:0', batch_size=16, num_samples=None):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = PairedImageDataset(ref_dir, gen_dir, transform=transform)
    
    if num_samples is not None:
        dataset.num_gen = min(num_samples, dataset.num_gen)
        dataset.gen_images = dataset.gen_images[:dataset.num_gen]
    
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()
    
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    
    print("Computing pairwise metrics (PSNR, SSIM, LPIPS)...")
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        
        ref_np = data['ref_np']
        gen_np = data['gen_np']
        ref_tensor = data['ref'].unsqueeze(0).to(device)
        gen_tensor = data['gen'].unsqueeze(0).to(device)
        
        # PSNR
        psnr_val = psnr(ref_np, gen_np, data_range=255)
        psnr_scores.append(psnr_val)
        
        # SSIM
        ssim_val = ssim(ref_np, gen_np, data_range=255, channel_axis=2)
        ssim_scores.append(ssim_val)
        
        # LPIPS
        ref_lpips = ref_tensor * 2 - 1
        gen_lpips = gen_tensor * 2 - 1
        with torch.no_grad():
            lpips_val = lpips_model(ref_lpips, gen_lpips).item()
        lpips_scores.append(lpips_val)
    
    return {
        'PSNR': np.mean(psnr_scores),
        'PSNR_std': np.std(psnr_scores),
        'SSIM': np.mean(ssim_scores),
        'SSIM_std': np.std(ssim_scores),
        'LPIPS': np.mean(lpips_scores),
        'LPIPS_std': np.std(lpips_scores),
    }


def compute_distribution_metrics(ref_dir, gen_dir, device='cuda:0', batch_size=32, num_samples=None):
    import tempfile, shutil

    ref_images = sorted([f for f in os.listdir(ref_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    gen_images = sorted([f for f in os.listdir(gen_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if num_samples is not None and num_samples < len(gen_images):
        gen_images = gen_images[:num_samples]
        tmp_gen = tempfile.mkdtemp()
        for fname in gen_images:
            src = os.path.join(gen_dir, fname)
            dst = os.path.join(tmp_gen, fname)
            shutil.copy2(src, dst)
        gen_dir_use = tmp_gen
    else:
        gen_dir_use = gen_dir
        tmp_gen = None

    torch_device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Computing Clean-FID with {len(ref_images)} ref and {len(gen_images)} gen images...")

    fid_score = clean_fid.compute_fid(ref_dir, gen_dir_use, mode='clean', device=torch_device, batch_size=batch_size, verbose=True)
    kid_score = clean_fid.compute_kid(ref_dir, gen_dir_use, mode='clean', device=torch_device, batch_size=batch_size, verbose=True)

    if tmp_gen is not None:
        shutil.rmtree(tmp_gen)

    return {
        'FID': fid_score,
        'KID': kid_score,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate image generation metrics')
    parser.add_argument('--ref_dir', type=str, required=True, help='Reference images directory')
    parser.add_argument('--gen_dir', type=str, required=True, help='Generated images directory')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for FID/KID')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to use (default: all)')
    parser.add_argument('--skip_pairwise', action='store_true', help='Skip pairwise metrics (PSNR, SSIM, LPIPS)')
    parser.add_argument('--skip_distribution', action='store_true', help='Skip distribution metrics (FID, KID)')
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    results = {}
    
    if not args.skip_pairwise:
        pairwise_results = compute_pairwise_metrics(
            args.ref_dir, args.gen_dir, 
            device=device, 
            num_samples=args.num_samples
        )
        results.update(pairwise_results)
    
    if not args.skip_distribution:
        dist_results = compute_distribution_metrics(
            args.ref_dir, args.gen_dir, 
            device=device, 
            batch_size=args.batch_size,
            num_samples=args.num_samples
        )
        results.update(dist_results)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    if 'PSNR' in results:
        print(f"PSNR:  {results['PSNR']:.4f} ± {results['PSNR_std']:.4f}")
        print(f"SSIM:  {results['SSIM']:.4f} ± {results['SSIM_std']:.4f}")
        print(f"LPIPS: {results['LPIPS']:.4f} ± {results['LPIPS_std']:.4f}")
    
    if 'FID' in results:
        print(f"FID:   {results['FID']:.4f}")
        print(f"KID:   {results['KID']:.6f}")
    
    print("="*50)
    
    return results


if __name__ == '__main__':
    main()
