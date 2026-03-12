try:
    import argparse
    import os
    import torch
    from pipelines.cog_pipeline import CogVideoXImageToVideoCGGPipeline
    from diffusers.utils import export_to_video, load_image
    from pathlib import Path
    from multiprocessing import Process,Value
    from pipelines.utils import split_list_evenly
except ImportError as e:
    raise ImportError(f"Required module not found: {e}. Please install it before running this script. "
                     f"For installation instructions, see: https://github.com/zai-org/CogVideo")

MODEL_ID = "THUDM/CogVideoX-5b-I2V"
DTYPE = torch.bfloat16

def parse_args():
    parser = argparse.ArgumentParser(description="Run Wan Image to Video Pipeline")
    parser.add_argument("--input_dir", type=str, default="data/example_cog/", help="Path to input images")
    parser.add_argument("--output_dir", type=str, default="outputs/cog/", help="Path to save output video")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--num-frames", type=int, default=49, help="Number of frames to generate")
    parser.add_argument("--guidance-scale", type=float, default=6.0, help="Guidance scale for generation")
    parser.add_argument("--alpha_1", type=int, default=4, help="Generated Samples with baseline")
    parser.add_argument("--alpha_2", type=int, default=8, help="Generated Samples with baseline")
    return parser.parse_args()


args = parse_args()


input_dir = args.input_dir

output_dir = args.output_dir + str(args.alpha_1) + "_" + str(args.alpha_2)+"/"

num_inference_steps = args.num_inference_steps
seed = args.seed
num_frames = args.num_frames
guidance_scale = args.guidance_scale



# make sure output directory exists
Path(os.path.dirname(output_dir) or ".").mkdir(parents=True, exist_ok=True)

# -----------------------
# Setup Pipeline
# -----------------------
def get_available_gpus_pytorch():
    available_gpus = []
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            available_gpus.append(i)
    return available_gpus

def setup_cog_pipeline(model_id: str, dtype: torch.dtype, device):
    pipe = CogVideoXImageToVideoCGGPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        # device_map="cuda",      # keep this
    ).to(device)
    pipe.vae.enable_tiling() # pipe.enable_vae_slicing()
    pipe.vae.enable_slicing() # pipe.enable_vae_tiling()
    # pipe = pipe.to(device)
    return pipe


def each_split(split_ls, device):
    pipe = setup_cog_pipeline(MODEL_ID, DTYPE, device)
    for (image_path, motion_signal_mask_path, motion_signal_video_path, prompt_path, output_path) in split_ls:
        image = load_image(image_path)
        # Load prompt (unchanged)
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        prompt = (prompt)

        # Generator / seed (unchanged)
        gen_device = device if device.startswith("cuda") else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(seed)
        
        with torch.inference_mode():
            result = pipe(
                        [image],
                        [prompt],
                        generator=generator,
                        num_inference_steps=num_inference_steps,
                        motion_signal_video_path=motion_signal_video_path,
                        motion_signal_mask_path=motion_signal_mask_path,
                        num_frames=num_frames,
                        guidance_scale=guidance_scale,
                        alpha_1=args.alpha_1,
                        alpha_2=args.alpha_2,
                        )

        frames = result.frames[0]
        Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)
        export_to_video(frames, output_path, fps=30)
        print(f"Video saved to {output_path}")


# -----------------------
# Main (same functional steps)
# -----------------------
def main():
    
    device_ls = get_available_gpus_pytorch()
    p = Path(input_dir)
    sub_dirs = [x.name for x in p.iterdir() if x.is_dir()]
    input_ls = []
    for each_sub_dir in sub_dirs:
        image_path = os.path.join(input_dir, each_sub_dir, "gt.png")
        motion_signal_mask_path = os.path.join(input_dir, each_sub_dir, "mask.mp4")
        motion_signal_video_path = os.path.join(input_dir, each_sub_dir, "reference.mp4")
        prompt_path = os.path.join(input_dir, each_sub_dir, "prompt.txt")
        output_path = os.path.join(output_dir, each_sub_dir+".mp4")
        if os.path.exists(output_path):
            continue
        input_ls.append((image_path, motion_signal_mask_path, motion_signal_video_path, prompt_path, output_path))
    
    chunks = split_list_evenly(input_ls, len(device_ls))
    
    process_list = []
    for i, device_id in enumerate(device_ls):
        p = Process(target=each_split, args=(chunks[i], "cuda:"+str(device_id)))
        p.start()
        process_list.append(p)
    for each in process_list:
        each.join()
    

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()