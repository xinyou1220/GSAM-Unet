# GATSUN — 冠狀血管語意分割（ARCADE）

本專案在靜態 X 光血管影像上，結合 **SAM3 視覺骨幹**、**UNet 解碼器**、**語意提示（Semantic Prompt）**、**稀疏圖注意力（SparseGAT）** 與 **Re-ID（InfoNCE）**，並支援 **貝茲曲線偽影增強** 與可選的 **Mean Teacher 半監督**（需無標籤資料路徑）。

---

## 專案重點

| 項目 | 說明 |
|------|------|
| 資料 | ARCADE 風格目錄；血管類別 **LAD / LCx / RCA**（對應 `vessel_id` 0/1/2） |
| 骨幹 | 自 `sam3_weights/sam3.pt` 載入 SAM3，decoder 等為可訓練 |
| 損失 | **Focal Tversky** + **InfoNCE**（λ_reid 可調） |
| 圖模組 | 預設 **SparseGAT**（2 層、4 head、k=16）；無 PyG 時可退為密集 GNN |
| 增強 | 訓練時以機率注入導管／導絲／胸骨縫線等 **偽影**（不污染 mask） |

核心實作：`unet_v2.py`（模型）、`train_v2.py`、`evaluate_v2.py`、`data_loader_v2.py`。上游 SAM3 原始碼在 `sam3git/`。

---

## 環境需求

以下為一次實際跑通記錄（見專案根目錄 `log.txt`）：

- **Python**：建議與 PyTorch 官方 wheel 對應的版本  
- **PyTorch**：`2.9.1+cu128`，**CUDA** 可用（日誌為 NVIDIA GeForce RTX 5090）  
- **PyTorch Geometric**：`2.7.0`（未安裝時會自動用密集 GNN 後備）  
- 其他腳本會檢查：**SciPy**、**OpenCV**、**scikit-image**  
- 需能 `import sam3`（透過安裝 `sam3git` 套件或將路徑加入 `PYTHONPATH`）

請自行準備與 CUDA 版本相符的 PyTorch / PyG 安裝指令；本 repo 未附 `requirements.txt`。

---

## 資料與目錄約定

- `--data` 指向資料根目錄（範例：`C:/Users/user/Desktop/ARCADE`）。  
- 載入邏輯見 `data_loader_v2.py`（含快取、合併重切 `merge_split`、偽影增強等）。  
- **SAM3 預訓練權重**：預設腳本使用 `sam3_weights/sam3.pt`（請自行放置對應檔案）。

---

## 一鍵訓練 + 評估

編輯 `run_all_v2.sh` 頂部變數後，在 **Git Bash**（或相容的 `sh`）執行：

```bash
sh run_all_v2.sh
```

Windows PowerShell 範例（與 `log.txt` 相同）：

```powershell
& "C:\Program Files\Git\bin\sh.exe" run_all_v2.sh
```

腳本會依設定：

1. 檢查 PyTorch / SAM3 / SciPy / OpenCV / skimage  
2. 呼叫 `train_v2.py` → 權重與日誌寫入 `checkpoints_v2/`  
3. 呼叫 `evaluate_v2.py` → 圖表與指標寫入 `results_v2/`

可切換 **`PER_VESSEL=true`** 改為三條血管各訓練一個模型（輸出子目錄 `checkpoints_v2/LAD` 等）。

---

## 手動指令範例

訓練（參數請與 `run_all_v2.sh` 內 `build_train_cmd` 對齊）：

```bash
python train_v2.py --data <ARCADE_ROOT> --vessels LAD,LCx,RCA \
  --img_size 512 --epochs 500 --batch 8 --accum_steps 1 --lr 1e-4 \
  --save_dir ./checkpoints_v2 --checkpoint <path/to/sam3.pt> \
  --merge_split --val_ratio 0.1 --split_seed 42 --amp \
  # ... 其餘 GAT / ReID / Tversky / artifact 等參數見 train_v2.py --help
```

評估：

```bash
python evaluate_v2.py --data <ARCADE_ROOT> \
  --ckpt ./checkpoints_v2/best_model.pth \
  --log_csv ./checkpoints_v2/train_log.csv \
  --vessels LAD,LCx,RCA --img_size 512 --batch 8 \
  --out_dir ./results_v2 --n_vis 20 --top_k 4 \
  --cldice_iter 10 --gnn_iters 3 --pp_min_size 50
```

---

## 實驗設定摘要

| 設定 | 值 |
|------|-----|
| 模式 | **JOINT**：單一模型，血管 `LAD,LCx,RCA` |
| 影像尺寸 | 512 |
| Epochs / Batch | 500 / 8（accum_steps=1） |
| 學習率 | 1e-4（backbone 縮放 0.01）；warmup 5 epoch |
| 資料切分 | `merge_split`，val_ratio=0.1，seed=42 |
| 樣本數（該次 run） | train+val 預載 1200 張 → train **1081** / val **119** |
| Semantic Prompt / SparseGAT / ReID | 開 |
| SparseGAT | 2L, 4H, k=16, max_nodes=4096, node_threshold=0.3 |
| GNN iters | 3 |
| Tversky | α=0.5, β=0.5, γ=4/3 |
| λ_reid | 0.1 |
| 偽影機率 `artifact_prob` | 0.45 |
| Mean Teacher | **OFF**（`UNLABELED_DIR` 為空） |
| AMP | ON |

SAM3 載入：`Loaded 442, missing 0, unexpected 22`。  
可訓練參數約 **2.97M**（總參數約 457M）。

---

## 訓練結果

- **總訓練時間**：約 **19 小時 9 分**（500 epoch，約 137–223 s/epoch 區間）。  
- **最佳驗證 Dice**：**0.8280**（`best_model.pth` 存於 `checkpoints_v2/`）。  
- 最後一個 epoch：`tr` Dice 係數約 **0.8616**，`va` 約 **0.8245**。  
- 完整逐 epoch 曲線見 `checkpoints_v2/train_log.csv`（欄位含 `tr_total`, `va_dice_coeff`, `lr` 等）。

---

## 測試集與閾值掃描

評估使用 **300** 張測試樣本。後處理預設：`min_size=50`，並以驗證集掃描得到的高閾值做推論（日誌中 **high=0.65, low=None**）。

### 主要測試指標（與驗證選閾一致時）

| 指標 | 數值 |
|------|------|
| Dice（mean ± std） | **0.7832 ± 0.1086** |
| clDice（mean ± std） | **0.7786 ± 0.1291** |
| LAD（n=87） | Dice **0.7445**, clDice **0.7470** |
| LCx（n=113） | Dice **0.7612**, clDice **0.7537** |
| RCA（n=100） | Dice **0.8416**, clDice **0.8341** |

### 驗證集閾值掃描（200 張）

- 最佳組合：**high=0.65, low=None**，Dice **0.8661**（Precision/Recall 約 0.8632 / 0.8690）。

### 測試集閾值掃描

- 在測試上最佳：**high=0.30, low=0.12**，Dice **0.7959**。  
- 日誌提示：驗證與測試的最佳 **high** 相差 **0.35**，代表 **val/test 分佈有差異**；若要在測試上極大化 Dice，宜在測試或獨立校準集上選閾值，而非直接沿用驗證集 high=0.65。

---

## 視覺化與結果檔案

執行 `evaluate_v2.py` 後，預期在 `results_v2/` 產生（路徑與 `log.txt` 一致）：

| 檔案 | 說明 |
|------|------|
| `training_curves.png` | 由 `train_log.csv` 繪製訓練曲線 |
| `reid_tsne.png` | 300 筆測試嵌入的 **t-SNE** |
| `summary_grid.png` | 視覺化摘要網格（原圖／預測／疊加 等） |
| `test_metrics.csv` | 每張測試圖的 **Dice / clDice / vessel_id** |

> **備註**：若你 clone 的目錄裡目前只有 `test_metrics.csv`，代表圖檔可能未一併提交；在本機重新跑一次 `evaluate_v2.py` 即可還原上述 PNG。

---

## `test_metrics.csv` 範例列

`vessel_id`：**0=LAD，1=LCx，2=RCA**。

**較佳範例（Dice 高）**

| filename | dice | cldice | vessel_id |
|----------|------|--------|-----------|
| 101 | 0.9493 | 0.9605 | 2 (RCA) |
| 195 | 0.9538 | 0.9873 | 2 (RCA) |
| 139 | 0.9370 | 0.9903 | 2 (RCA) |
| 291 | 0.9055 | 0.9734 | 0 (LAD) |

**較難範例（Dice 低，可供錯誤分析）**

| filename | dice | cldice | vessel_id |
|----------|------|--------|-----------|
| 88 | 0.2104 | 0.1648 | 0 (LAD) |
| 72 | 0.3240 | 0.1788 | 0 (LAD) |
| 142 | 0.3034 | 0.2507 | 0 (LAD) |

完整 300 筆見 `results_v2/test_metrics.csv`。

---

## 常見調整

- **半監督**：在 `run_all_v2.sh` 設定 `UNLABELED_DIR`，並確認 `consist_max_weight`、`consist_ramp_epochs`、`pseudo_threshold`。  
- **關閉 Re-ID 或語意提示**：`train_v2.py` 的 `--no_reid`、`--no_semantic_prompt`。  
- **密集 GNN**：`--use_dense_gnn`（取代 SparseGAT）。  
- **跳過評估階段閾值掃描**（加快）：`evaluate_v2.py --skip_thr_sweep`。

---

## 授權與引用

使用 `sam3git/` 內容時請遵循原 SAM3 專案授權與論文引用要求；本 README 僅描述本目錄內之訓練與評估流程。
