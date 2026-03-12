# for CogVideoX
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_cog.py \
    --input_dir data/datasets_cog/ \
    --output_dir outputs/cog/ \
    --alpha_1 4 \
    --alpha_2 8 \

python evaluation.py \
    --target_method cog/4_8 \
    --root_path datasets_cog/


# for Wan2.2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_wan.py \
    --input_dir data/datasets_wan/ \
    --output_dir outputs/wan/ \
    --alpha_1 16 \
    --alpha_2 20 \

python evaluation.py \
    --target_method wan/16_20 \
    --root_path datasets_wan/

