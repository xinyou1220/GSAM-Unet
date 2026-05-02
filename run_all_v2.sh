set -euo pipefail


DATA_ROOT="ARCADE"
SAM3_CHECKPOINT="sam3_weights/sam3.pt"
SAVE_DIR="./checkpoints_v2"
RESULT_DIR="./results_v2"
VESSELS="LAD,LCx,RCA"

# 無標籤資料路徑（留空 = 純監督式訓練）
UNLABELED_DIR=""

PER_VESSEL=false

MERGE_SPLIT=true
VAL_RATIO=0.1
SPLIT_SEED=42


IMG_SIZE=512
EPOCHS=500
BATCH=8
ACCUM_STEPS=1
LR=1e-4
SCALE_LR=false
WORKERS=0
SAVE_EVERY=10
RESUME=""


WARMUP_EPOCHS=5


UNFREEZE=false
BACKBONE_LR_SCALE=0.01

USE_SEMANTIC_PROMPT=true
USE_SPARSE_GAT=true
USE_REID=true

GAT_LAYERS=2
GAT_HEADS=4
K_NEIGHBORS=16
MAX_NODES=4096
NODE_THRESHOLD=0.3

GNN_ITERS=3


N_PROMPT_TOKENS=8
REID_EMBED_DIM=128
LAMBDA_REID=0.1


ARTIFACT_PROB=0.45           # 訓練資料帶偽影（導管/導絲/縫線）


CONSIST_MAX_WEIGHT=1.0       # consistency loss 最大權重
CONSIST_RAMP_EPOCHS=20       # 前 N 個 epoch 線性 ramp-up
PSEUDO_THRESHOLD=0.7         # Teacher pseudo label 信心門檻

TVERSKY_ALPHA=0.5
TVERSKY_BETA=0.5
TVERSKY_GAMMA=1.3333

CLDICE_ITER=10
PP_MIN_SIZE=50
SKIP_THR_SWEEP=false
N_VIS=20
TOP_K=4


USE_AMP=true
USE_COMPILE=false
NO_CACHE=false

echo "================================================"
echo " SAM3 UNet V2"
echo " SemanticPrompt + SparseGAT + ReID"
echo " + Bezier Artifact Aug + Mean Teacher"
echo "================================================"
python -c "
import torch, sam3
print(f'PyTorch : {torch.__version__}')
print(f'CUDA    : {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"})')
try:
    import torch_geometric
    print(f'PyG     : {torch_geometric.__version__}')
except ImportError:
    print('PyG     : NOT INSTALLED (dense GNN fallback)')
" 2>/dev/null || { echo "[ERROR] PyTorch/SAM3 not found."; exit 1; }
python -c "import scipy"   2>/dev/null || { echo "[ERROR] SciPy not found."; exit 1; }
python -c "import cv2"     2>/dev/null || { echo "[ERROR] opencv-python not found."; exit 1; }
python -c "import skimage" 2>/dev/null || { echo "[ERROR] scikit-image not found."; exit 1; }

echo ""
echo "Settings:"
echo "  PER_VESSEL=${PER_VESSEL}  MERGE_SPLIT=${MERGE_SPLIT}"
echo "  EPOCHS=${EPOCHS}  BATCH=${BATCH}x${ACCUM_STEPS}  LR=${LR}"
echo "  SemanticPrompt=${USE_SEMANTIC_PROMPT}  ReID=${USE_REID}"
echo "  SparseGAT=${USE_SPARSE_GAT} (${GAT_LAYERS}L ${GAT_HEADS}H k=${K_NEIGHBORS})"
echo "  Tversky(α=${TVERSKY_ALPHA}, β=${TVERSKY_BETA}, γ=${TVERSKY_GAMMA})"
echo "  λ_reid=${LAMBDA_REID}  artifact=${ARTIFACT_PROB}"
if [[ -n "${UNLABELED_DIR}" ]]; then
    echo "  MeanTeacher=ON (ramp=${CONSIST_RAMP_EPOCHS} w=${CONSIST_MAX_WEIGHT} thr=${PSEUDO_THRESHOLD})"
else
    echo "  MeanTeacher=OFF"
fi
echo ""


build_train_cmd() {
    local VESSEL_ARG="$1"
    local SAVE="$2"

    local CMD="python train_v2.py"
    CMD+=" --data ${DATA_ROOT}"
    CMD+=" --vessels ${VESSEL_ARG}"
    CMD+=" --img_size ${IMG_SIZE}"
    CMD+=" --epochs ${EPOCHS}"
    CMD+=" --batch ${BATCH}"
    CMD+=" --accum_steps ${ACCUM_STEPS}"
    CMD+=" --lr ${LR}"
    CMD+=" --workers ${WORKERS}"
    CMD+=" --save_dir ${SAVE}"
    CMD+=" --save_every ${SAVE_EVERY}"
    CMD+=" --warmup_epochs ${WARMUP_EPOCHS}"
    CMD+=" --backbone_lr_scale ${BACKBONE_LR_SCALE}"
    CMD+=" --tversky_alpha ${TVERSKY_ALPHA}"
    CMD+=" --tversky_beta ${TVERSKY_BETA}"
    CMD+=" --tversky_gamma ${TVERSKY_GAMMA}"
    CMD+=" --val_ratio ${VAL_RATIO}"
    CMD+=" --split_seed ${SPLIT_SEED}"
    CMD+=" --n_prompt_tokens ${N_PROMPT_TOKENS}"
    CMD+=" --reid_embed_dim ${REID_EMBED_DIM}"
    CMD+=" --lambda_reid ${LAMBDA_REID}"
    CMD+=" --gat_layers ${GAT_LAYERS}"
    CMD+=" --gat_heads ${GAT_HEADS}"
    CMD+=" --k_neighbors ${K_NEIGHBORS}"
    CMD+=" --max_nodes ${MAX_NODES}"
    CMD+=" --node_threshold ${NODE_THRESHOLD}"
    CMD+=" --gnn_iters ${GNN_ITERS}"
    CMD+=" --artifact_prob ${ARTIFACT_PROB}"
    CMD+=" --consist_max_weight ${CONSIST_MAX_WEIGHT}"
    CMD+=" --consist_ramp_epochs ${CONSIST_RAMP_EPOCHS}"
    CMD+=" --pseudo_threshold ${PSEUDO_THRESHOLD}"

    [[ "${USE_AMP}"             == true  ]] && CMD+=" --amp"
    [[ "${USE_COMPILE}"         == true  ]] && CMD+=" --compile"
    [[ "${SCALE_LR}"            == true  ]] && CMD+=" --scale_lr"
    [[ "${NO_CACHE}"            == true  ]] && CMD+=" --no_cache"
    [[ "${UNFREEZE}"            == true  ]] && CMD+=" --unfreeze"
    [[ "${MERGE_SPLIT}"         == true  ]] && CMD+=" --merge_split"
    [[ "${USE_SEMANTIC_PROMPT}" == false ]] && CMD+=" --no_semantic_prompt"
    [[ "${USE_SPARSE_GAT}"      == false ]] && CMD+=" --use_dense_gnn"
    [[ "${USE_REID}"            == false ]] && CMD+=" --no_reid"
    [[ -n "${SAM3_CHECKPOINT}"  ]]         && CMD+=" --checkpoint ${SAM3_CHECKPOINT}"
    [[ -n "${RESUME}"           ]]         && CMD+=" --resume ${RESUME}"
    [[ -n "${UNLABELED_DIR}"    ]]         && CMD+=" --unlabeled_dir ${UNLABELED_DIR}"

    echo "${CMD}"
}


build_eval_cmd() {
    local VESSEL_ARG="$1"
    local SAVE="$2"
    local RESULT="$3"

    local CMD="python evaluate_v2.py"
    CMD+=" --data ${DATA_ROOT}"
    CMD+=" --ckpt ${SAVE}/best_model.pth"
    CMD+=" --log_csv ${SAVE}/train_log.csv"
    CMD+=" --vessels ${VESSEL_ARG}"
    CMD+=" --img_size ${IMG_SIZE}"
    CMD+=" --batch ${BATCH}"
    CMD+=" --workers ${WORKERS}"
    CMD+=" --out_dir ${RESULT}"
    CMD+=" --n_vis ${N_VIS}"
    CMD+=" --top_k ${TOP_K}"
    CMD+=" --cldice_iter ${CLDICE_ITER}"
    CMD+=" --gnn_iters ${GNN_ITERS}"
    CMD+=" --pp_min_size ${PP_MIN_SIZE}"

    [[ "${NO_CACHE}"        == true ]] && CMD+=" --no_cache"
    [[ "${SKIP_THR_SWEEP}"  == true ]] && CMD+=" --skip_thr_sweep"

    echo "${CMD}"
}


run_one() {
    local LABEL="$1"
    local VESSEL_ARG="$2"
    local SAVE="$3"
    local RESULT="$4"

    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "  Training : ${LABEL}  (vessels=${VESSEL_ARG})"
    echo "╚══════════════════════════════════════════════╝"

    local TCMD; TCMD=$(build_train_cmd "${VESSEL_ARG}" "${SAVE}")
    echo "Command: ${TCMD}"
    echo ""
    eval "${TCMD}"

    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "  Evaluating : ${LABEL}"
    echo "╚══════════════════════════════════════════════╝"

    local ECMD; ECMD=$(build_eval_cmd "${VESSEL_ARG}" "${SAVE}" "${RESULT}")
    echo "Command: ${ECMD}"
    echo ""
    eval "${ECMD}"

    echo "  ✓ ${LABEL} done.  ckpt → ${SAVE}  results → ${RESULT}"
}

if [[ "${PER_VESSEL}" == true ]]; then
    echo "Mode: PER-VESSEL (3 models)"
    for V in LAD LCx RCA; do
        run_one "${V}" "${V}" "${SAVE_DIR}/${V}" "${RESULT_DIR}/${V}"
    done

    echo ""
    echo "============================================"
    echo " All per-vessel runs complete"
    echo "============================================"
    for V in LAD LCx RCA; do
        CSV="${RESULT_DIR}/${V}/test_metrics.csv"
        if [[ -f "${CSV}" ]]; then
            python -c "
import csv, statistics
with open('${CSV}') as f:
    rows = list(csv.DictReader(f))
dices  = [float(r['dice'])   for r in rows]
clds   = [float(r['cldice']) for r in rows]
n      = len(dices)
d_std  = statistics.stdev(dices)  if n > 1 else 0.0
cl_std = statistics.stdev(clds)   if n > 1 else 0.0
print(f'  ${V:>3}: n={n:3d}  '
      f'Dice={statistics.mean(dices):.4f}+/-{d_std:.4f}  '
      f'clDice={statistics.mean(clds):.4f}+/-{cl_std:.4f}')
" 2>/dev/null || echo "  ${V}: (no results)"
        else
            echo "  ${V}: (no CSV found)"
        fi
    done
else
    echo "Mode: JOINT (1 model)"
    run_one "ALL" "${VESSELS}" "${SAVE_DIR}" "${RESULT_DIR}"
fi

echo ""
echo "============================================"
echo " Done."
echo "============================================"
