# for CogVideoX
CUDA_VISIBLE_DEVICES=0 python run_cog.py \
    --output_dir outputs/cog/ \
    --w1 4 \
    --w2 8 \

# for Wan2.2
CUDA_VISIBLE_DEVICES=0 python run_wan.py \
    --output_dir outputs/wan/ \
    --w1 16 \
    --w2 20 \

