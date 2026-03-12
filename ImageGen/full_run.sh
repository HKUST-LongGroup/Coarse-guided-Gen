set -e

ALPHA=5
TASKS=("motion_blur" "gaussian_blur" "super_resolution" "inpainting_random")


# ── Parse arguments ───────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --alpha)          ALPHA="$2";          shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Build extra flags to pass to sample_condition.py ─────────────
EXTRA_FLAGS="--multi_gpu"
EXTRA_FLAGS="$EXTRA_FLAGS --alpha $ALPHA"


# ── Determine save_dir (mirrors logic in sample_condition.py) ─────
SAVE_DIR="./outputs/our_alpha_${ALPHA}"

# ── Result / log paths ────────────────────────────────────────────
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${SAVE_DIR}/eval_results_${TIMESTAMP}.txt"

echo "=================================================="
echo "  Run configuration"
echo "    alpha          : $ALPHA"
echo "    save_dir       : $SAVE_DIR"
echo "    log_file       : $LOG_FILE"
echo "=================================================="

# ── Write log header ──────────────────────────────────────────────
mkdir -p "$SAVE_DIR"
{
    echo "Run timestamp : $TIMESTAMP"
    echo "alpha         : $ALPHA"
    echo "save_dir      : $SAVE_DIR"
    echo ""
} > "$LOG_FILE"

# ── Task list ─────────────────────────────────────────────────────
declare -A TASK_CONFIGS=(
    [gaussian_blur]="configs/gaussian_deblur_config.yaml"
    [motion_blur]="configs/motion_deblur_config.yaml"
    [super_resolution]="configs/super_resolution_config.yaml"
    [inpainting_random]="configs/inpainting_random_config.yaml"
)


# ── Run sampling for each task ────────────────────────────────────
for TASK in "${TASKS[@]}"; do


    CONFIG="${TASK_CONFIGS[$TASK]}"
    echo ""
    echo ">>> [$(date +"%H:%M:%S")] Sampling: $TASK"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 sample_condition.py \
        --model_config=configs/model_config.yaml \
        --diffusion_config=configs/diffusion_config.yaml \
        --task_config="$CONFIG" \
        $EXTRA_FLAGS
    echo "    Done: $TASK"



    echo ""
    echo ">>> Starting evaluation..."
    echo "=================================================" >> "$LOG_FILE"
    echo "EVALUATION RESULTS" >> "$LOG_FILE"
    echo "=================================================" >> "$LOG_FILE"

    REF_DIR="${SAVE_DIR}/${TASK}/label/"
    GEN_DIR="${SAVE_DIR}/${TASK}/recon/"

    if [[ ! -d "$GEN_DIR" ]]; then
        echo "    [WARN] $GEN_DIR not found, skipping $TASK"
        continue
    fi

    echo ""
    echo ">>> [$(date +"%H:%M:%S")] Evaluating: $TASK"
    echo "" >> "$LOG_FILE"
    echo "--- $TASK ---" >> "$LOG_FILE"

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python3 evaluate.py \
        --ref_dir "$REF_DIR" \
        --gen_dir "$GEN_DIR" \
        --device cuda \
        --batch_size 32 \
        2>&1 | tee -a "$LOG_FILE"
done


echo ""
echo "=================================================="
echo "  All done. Results saved to: $LOG_FILE"
echo "=================================================="
