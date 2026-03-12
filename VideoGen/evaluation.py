import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.models.video import r3d_18, R3D_18_Weights
import lpips
import clip
from PIL import Image
from scipy import linalg
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
import argparse
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_video_frames(video_path, max_frames=None):
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return []
        
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames

def calculate_mse(img1, img2):
    img1_norm = img1.astype(np.float32) / 255.0
    img2_norm = img2.astype(np.float32) / 255.0
    return np.mean((img1_norm - img2_norm) ** 2)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of Frechet Distance"""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ("fid calculation produces singular product; "
               "adding %s to diagonal of cov estimates") % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


class Evaluator:
    def __init__(self):
        print(f"Initializing Evaluator on {device}...")

        # 1. LPIPS
        print(" -> Loading LPIPS model...")
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        self.lpips_fn.eval()

        # 2. RAFT
        print(" -> Loading RAFT model...")
        weights = Raft_Large_Weights.DEFAULT
        self.raft_model = raft_large(weights=weights).to(device)
        self.raft_model.eval()
        self.raft_transform = weights.transforms()

        # 3. CLIP
        print(" -> Loading CLIP model...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()
        
        # 4. FVD Model (R3D-18)
        print(" -> Initializing FVD feature extractor (R3D-18)...")
        weights_r3d = R3D_18_Weights.DEFAULT
        base_r3d = r3d_18(weights=weights_r3d)

        self.fvd_model = nn.Sequential(*list(base_r3d.children())[:-1]) 
        self.fvd_model.to(device)
        self.fvd_model.eval()
        self.fvd_transform = weights_r3d.transforms()
        
        self.fvd_feats_real = []
        self.fvd_feats_gen = []

        print(" -> Loading DINOv2 model...")
        self.dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
        self.dinov2_model.eval()
        
        self.dinov2_transform = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image_for_lpips(self, img_np):
        img = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img = img * 2 - 1
        return img.to(device)
    
    def get_raft_flow(self, img1_np, img2_np):
        img1 = torch.from_numpy(img1_np).permute(2, 0, 1)
        img2 = torch.from_numpy(img2_np).permute(2, 0, 1)
        
        img1, img2 = self.raft_transform(img1, img2)
        img1 = img1.unsqueeze(0).to(device)
        img2 = img2.unsqueeze(0).to(device)
        
        n, c, h, w = img1.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            img1 = F.pad(img1, (0, pad_w, 0, pad_h), mode='replicate')
            img2 = F.pad(img2, (0, pad_w, 0, pad_h), mode='replicate')
        
        with torch.no_grad():
            list_of_flows = self.raft_model(img1, img2)
            predicted_flow = list_of_flows[-1]
        
        if pad_h > 0 or pad_w > 0:
            predicted_flow = predicted_flow[:, :, :h, :w]
        
        return predicted_flow

    def get_clip_embedding(self, img_np):
        img_pil = Image.fromarray(img_np)
        img_tensor = self.clip_preprocess(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(img_tensor)
        return image_features / image_features.norm(dim=-1, keepdim=True)
    
    def calculate_dinov2_distance(self, img1_np, img2_np):
        img1 = self.dinov2_transform(img1_np).unsqueeze(0).to(device)
        img2 = self.dinov2_transform(img2_np).unsqueeze(0).to(device)
        
        with torch.no_grad():
            feat1 = self.dinov2_model(img1)
            feat2 = self.dinov2_model(img2)
            
            feat1 = F.normalize(feat1, dim=-1)
            feat2 = F.normalize(feat2, dim=-1)
            
            similarity = (feat1 @ feat2.T).item()
            
        return 1.0 - similarity


    def accumulate_fvd_features(self, frames_gt, frames_gen):
        """
        """
        def process_video_clip(frames_list):
            vid_np = np.stack(frames_list)
            vid = torch.from_numpy(vid_np).permute(0, 3, 1, 2)
            vid = self.fvd_transform(vid)
        
            
            if vid.shape[0] != 3 and vid.shape[1] == 3:
                vid = vid.permute(1, 0, 2, 3)
            elif vid.shape[0] == 3:
                pass
            else:
                pass
            
            # 5. Add Batch -> (1, C, T, H, W)
            vid = vid.unsqueeze(0)
            vid = vid.to(device)
            
            with torch.no_grad():
                # Extract features -> (1, 512, 1, 1, 1)
                feats = self.fvd_model(vid)
                # Flatten -> (1, 512)
                feats = feats.view(feats.shape[0], -1).cpu().numpy()
            return feats

        if len(frames_gt) > 8: 
            self.fvd_feats_real.append(process_video_clip(frames_gt))
            self.fvd_feats_gen.append(process_video_clip(frames_gen))

    def compute_final_metrics(self):
        results = {}
        
        print("Computing final global FVD...")
        try:
            if len(self.fvd_feats_real) > 1 and len(self.fvd_feats_gen) > 1:
                # Concatenate all video features
                real_feats = np.concatenate(self.fvd_feats_real, axis=0)
                gen_feats = np.concatenate(self.fvd_feats_gen, axis=0)
                
                # Compute Stats
                mu_real = np.mean(real_feats, axis=0)
                sigma_real = np.cov(real_feats, rowvar=False)
                
                mu_gen = np.mean(gen_feats, axis=0)
                sigma_gen = np.cov(gen_feats, rowvar=False)
                
                # Compute Distance
                fvd_score = calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
                results['FVD'] = fvd_score
            else:
                print("Not enough videos for FVD calculation (need > 1).")
                results['FVD'] = -1.0
        except Exception as e:
            print(f"FVD Error: {e}")
            results['FVD'] = -1.0
            
        return results

def main_evaluation(video_pairs,output_path):
    if not video_pairs:
        print("No pairs to evaluate.")
        return

    evaluator = Evaluator()
    all_metrics = defaultdict(list)
    
    print(f"\nStarting evaluation loop for {len(video_pairs)} pairs...")
    
    for idx, (path_gen, path_gt) in enumerate(tqdm(video_pairs)):
        frames_gen = read_video_frames(path_gen)
        frames_gt = read_video_frames(path_gt)
        
        if not frames_gen or not frames_gt:
            continue

        n_frames = min(len(frames_gt), len(frames_gen))
        if n_frames < 2:
            continue

        frames_gt = frames_gt[:n_frames]
        frames_gen = frames_gen[:n_frames]
        
        evaluator.accumulate_fvd_features(frames_gt, frames_gen)
        
        vid_mse = []
        vid_lpips = []
        vid_dinov2_dist = []
        vid_flow_mse = []
        vid_clip_cons = []
        
        for i in range(n_frames):
            frame_gt = frames_gt[i]
            frame_gen = frames_gen[i]
            
            vid_mse.append(calculate_mse(frame_gt, frame_gen))
            
            # LPIPS
            t_gt = evaluator.preprocess_image_for_lpips(frame_gt)
            t_gen = evaluator.preprocess_image_for_lpips(frame_gen)
            with torch.no_grad():
                val = evaluator.lpips_fn(t_gt, t_gen)
            vid_lpips.append(val.item())

            # DINOv2
            dino_d = evaluator.calculate_dinov2_distance(frame_gt, frame_gen)
            vid_dinov2_dist.append(dino_d)
            
            if i < n_frames - 1:
                flow_gt = evaluator.get_raft_flow(frames_gt[i], frames_gt[i+1])
                flow_gen = evaluator.get_raft_flow(frames_gen[i], frames_gen[i+1])
                loss = F.mse_loss(flow_gt, flow_gen).item()
                vid_flow_mse.append(loss)
                
            if i < n_frames - 1:
                emb_curr = evaluator.get_clip_embedding(frames_gen[i])
                emb_next = evaluator.get_clip_embedding(frames_gen[i+1])
                sim = (emb_curr @ emb_next.T).item()
                vid_clip_cons.append(sim)

        all_metrics["MSE"].append(np.mean(vid_mse))
        all_metrics["LPIPS"].append(np.mean(vid_lpips))
        all_metrics["DINOv2_Dist"].append(np.mean(vid_dinov2_dist))
        if vid_flow_mse:
            all_metrics["Motion_Consistency"].append(np.mean(vid_flow_mse))
        if vid_clip_cons:
            all_metrics["Temporal_Consistency"].append(np.mean(vid_clip_cons))

    output_lines = []
    separator = "#"*60
    
    output_lines.append("\n" + separator)
    output_lines.append(separator)
    
    global_scores = evaluator.compute_final_metrics()
    for k, v in global_scores.items():
        output_lines.append(f"{k + ' (Global)':<30}: {v:.4f}")
        
    for k, v_list in all_metrics.items():
        if len(v_list) > 0:
            avg_val = np.mean(v_list)
            if "MSE" in k or "Dist" in k: 
                line = f"{k:<30}: {avg_val:.6f}" 
            else:
                line = f"{k:<30}: {avg_val:.4f}"
        else:
            line = f"{k:<30}: N/A"
        output_lines.append(line)
            
    output_lines.append(separator + "\n")
    
    for line in output_lines:
        print(line)
        
    os.makedirs(output_path, exist_ok=True)
    output_file = output_path + "/result.txt"
    with open(output_file, "w", encoding='utf-8') as f:
        for line in output_lines:
            f.write(line + "\n")



def parse_args():
    parser = argparse.ArgumentParser(description="Run Wan Image to Video Pipeline")
    parser.add_argument("--target_method", type=str)
    parser.add_argument("--root_path", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    target_method = args.target_method
    root_path = args.root_path
    video_pairs = []
    target_folder = Path(root_path)
    subdirs = [x for x in target_folder.iterdir() if x.is_dir()]
    output_path = "quant_results/"+target_method
    for folder in subdirs:
        base_path = root_path + folder.name + "/"
        if os.path.exists("outputs/" + target_method + "/" +folder.name+".mp4"):
            video_pairs.append(("outputs/" + target_method + "/" + folder.name+".mp4",base_path+"gt.mp4"))
    if not video_pairs:
        print("Please populate 'video_pairs_input' with actual paths.")
    else:
        main_evaluation(video_pairs,output_path)