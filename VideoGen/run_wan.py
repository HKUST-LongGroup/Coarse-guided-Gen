try:
    import os
    from pathlib import Path
    import torch
    from diffusers.utils import export_to_video, load_image
    from pipelines.wan_pipeline import WanImageToVideoCGGPipeline
    from pipelines.utils import (
        validate_inputs,
        compute_hw_from_area,
        split_list_evenly,
    )
    import argparse
    from multiprocessing import Process,Value
except ImportError as e:
    raise ImportError(f"Required module not found: {e}. Please install it before running this script. "
                     f"For installation instructions, see: https://github.com/Wan-Video/Wan2.2")

MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
DTYPE = torch.bfloat16

# -----------------------
# Argument Parser
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run Wan Image to Video Pipeline")
    parser.add_argument("--input_dir", type=str, default="data/example_wan/", help="Path to input images")
    parser.add_argument("--output_dir", type=str, default="outputs/wan/", help="Path to save output video")
    parser.add_argument("--negative-prompt", type=str, default=(
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
        "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
        "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    ), help="Default negative prompt in Wan2.2")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--max_area", type=int, default=832*480, help="Maximum area for resizing")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of frames to generate")
    parser.add_argument("--guidance-scale", type=float, default=3.5, help="Guidance scale for generation")
    parser.add_argument("--alpha_1", type=int, default=16, help="Generated Samples with baseline")
    parser.add_argument("--alpha_2", type=int, default=20, help="Generated Samples with baseline")
    return parser.parse_args()


args = parse_args()

input_dir = args.input_dir
output_dir = args.output_dir + str(args.alpha_1) + "_" + str(args.alpha_2)+"/"
negative_prompt = args.negative_prompt
num_inference_steps = args.num_inference_steps
seed = args.seed
max_area = args.max_area
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

def setup_wan_pipeline(model_id: str, dtype: torch.dtype, device: str):
    pipe = WanImageToVideoCGGPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.to(device)
    return pipe


def each_split(split_ls, device):
    pipe = setup_wan_pipeline(MODEL_ID, DTYPE, device)
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    for (image_path, motion_signal_mask_path, motion_signal_video_path, prompt_path, output_path) in split_ls:
        validate_inputs(image_path, motion_signal_mask_path, motion_signal_video_path)
        # Load and resize image (unchanged logic)
        image = load_image(image_path)
        height, width = compute_hw_from_area(image.height, image.width, max_area, mod_value)
        image = image.resize((width, height))
        # Load prompt (unchanged)
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        prompt = (prompt)

        # Generator / seed (unchanged)
        gen_device = device if device.startswith("cuda") else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(seed)

        with torch.inference_mode():
            result = pipe(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                motion_signal_video_path=motion_signal_video_path,
                motion_signal_mask_path=motion_signal_mask_path,
                alpha_1=args.alpha_1,
                alpha_2=args.alpha_2,
            )

        frames = result.frames[0]
        Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)
        export_to_video(frames, output_path, fps=16)
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