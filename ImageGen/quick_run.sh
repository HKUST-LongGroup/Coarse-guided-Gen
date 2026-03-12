set -e

ALPHA=5
TASKS=("super_resolution")


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
    [super_resolution]="configs/quick_start.yaml"
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



done

echo ""
echo "=================================================="
echo "  All done. Results saved to: $LOG_FILE"
echo "=================================================="
