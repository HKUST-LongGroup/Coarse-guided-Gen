# for CogVideoX
CUDA_VISIBLE_DEVICES=0 python run_cog.py \
    --output_dir outputs/cog/ \
    --alpha_1 4 \
    --alpha_2 8 \

# for Wan2.2
CUDA_VISIBLE_DEVICES=0 python run_wan.py \
    --output_dir outputs/wan/ \
    --alpha_1 16 \
    --alpha_2 20 \

