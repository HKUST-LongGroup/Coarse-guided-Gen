from functools import partial
import os
import argparse
import yaml

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main_worker(rank, world_size, args):
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{rank}" if args.multi_gpu else (f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')
    logger.info(f"Process {rank}: Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
   
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    if rank == 0:
        logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    if rank == 0:
        logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn, alpha=args.alpha)
   
    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    if measure_config['operator']['name'] == 'inpainting':
        out_path += f"_{measure_config['mask_opt']['mask_type']}"
    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    
    # Split dataset for this rank
    indices = list(range(rank, len(dataset), world_size))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, num_workers=8, shuffle=False)

    # Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )
        
    # Do Inference
    for step, ref_img in enumerate(loader):
        i = indices[step] # Original index
        logger.info(f"Process {rank}: Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)

        # Exception) In case of inpainging,
        if measure_config['operator'] ['name'] == 'inpainting':
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            measurement_cond_fn_cur = partial(cond_method.conditioning, mask=mask)
            sample_fn_cur = partial(sample_fn, measurement_cond_fn=measurement_cond_fn_cur)

            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)
        else: 
            sample_fn_cur = sample_fn
            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img)
            y_n = noiser(y)
         
        # Sampling
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        sample = sample_fn_cur(x_start=x_start, measurement=y_n, record=False, save_root=out_path)

        plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
        plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
        plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--multi_gpu', action='store_true', help='Use all available GPUs for parallel inference')
    parser.add_argument('--alpha', type=int, default=5)
    args = parser.parse_args()
    print(args)
    # Create directories before spawning processes
    task_config = load_yaml(args.task_config)
    args.save_dir = f'./outputs/our_alpha_{args.alpha}'
    measure_config = task_config['measurement']
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    if measure_config['operator']['name'] == 'inpainting':
        out_path += f"_{measure_config['mask_opt']['mask_type']}"
    
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    if args.multi_gpu:
        world_size = torch.cuda.device_count()
        print(f"Using {world_size} GPUs for parallel inference.")
        mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))
    else:
        main_worker(0, 1, args)

if __name__ == '__main__':
    main()
