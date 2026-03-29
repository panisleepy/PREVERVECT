# PREVERVECT

**PREVERVECT** 是一套以 **SpecXNet** 為核心的 Deepfake 偵測研究系統：結合 **空間域（RGB）** 與 **頻域（FFT 功率譜）** 雙流特徵、**DFA（雙流融合）**，並可搭配 **rPPG（遠端心率）** 與即時 **螢幕擷取偵測器**。本文件說明整體架構、計算與分析方法、以及各模組如何執行，方便在另一台電腦上閱讀與重現。

---

## 目錄

1. [系統架構總覽](#1-系統架構總覽)
2. [核心模型：SpecXNet](#2-核心模型specxnet)
3. [訓練流程與資料管線](#3-訓練流程與資料管線)
4. [評估指標與分析](#4-評估指標與分析)
5. [即時偵測器 `capture/screen_detector.py`](#5-即時偵測器-capturescreen_detectorpy)
6. [rPPG 研究管線](#6-rppg-研究管線)
7. [專案目錄結構](#7-專案目錄結構)
8. [環境與快速指令](#8-環境與快速指令)

---

## 1. 系統架構總覽

概念上分層如下（實作程度依模組而異）：

| 層級 | 角色 | 實作位置 / 說明 |
|------|------|-----------------|
| **影像淨化（MCDM）** | 對抗噪訊／淨化前處理 | `screen_detector` 中 `_mcdm_preprocess_placeholder` 為佔位，可替換為擴散或濾波模組。 |
| **雙流偵測 SpecXNet** | 臉部 RGB + 2D FFT 功率譜，經 DFA 融合後二元分類（Real / Fake） | `core/model.py`、`core/train.py` |
| **rPPG 生理線索** | 由臉部 ROI 的 RGB 時序估計脈動與參考 BPM | `screen_detector`（即時）、`advanced_extractor.py` 等（離線研究） |
| **NemoClaw / 延遲監控** | 主動防禦、延遲監控 | 即時 UI 顯示 **Latency (ms)**；其餘為架構預留。 |

**資料流（訓練）**：影片 → 抽幀 → 增強（可選）→ 臉部幀 BGR → RGB tensor + FFT 譜 tensor → SpecXNet → BCE 損失。

**資料流（即時偵測）**：螢幕擷取 → 人臉框 ROI → 預處理與銳化 → 雙流推論 → 分數校準與平滑 → UI；另以 RGB 序列做 rPPG 與 BPM 估計。

---

## 2. 核心模型：SpecXNet

### 2.1 雙流輸入

- **空間流**：臉部區域 **BGR** 轉 **RGB**，resize 至 **224×224**，**ImageNet** 標準化  
  \(\mu=(0.485,0.456,0.406)\), \(\sigma=(0.229,0.224,0.225)\)。
- **頻域流**：對同一幀 BGR 計算 **2D 功率譜（log 壓縮、DC 置中）**（`utils/fft_tools.py`），再 resize 至 224×224，線性縮放到 \([-1,1]\) 等價（0.5 / 0.5 normalize）。

### 2.2 骨干與 DFA

- **EfficientNet-B0**（`timm`）**兩路獨立**：一路吃 RGB，一路吃 FFT 譜圖；**全域平均池化**後得到兩個 \(D\) 維向量。
- **DFA（Dual-stream Feature Aggregation）**：將兩路特徵拼接後經 **MLP → softmax**，得到 **空間權重 \(w_s\)** 與 **頻域權重 \(w_f\)**（\(w_s+w_f=1\)），再 **加權融合** 為單一向量。
- **分類頭**：**Dropout** + **Linear(1)**，訓練時為 **BCEWithLogitsLoss**；推論時 **sigmoid** 為 **Fake 機率**。

### 2.3 推論時「畫質自適應」（偵測器）

當臉部 ROI 的 **Laplacian 變異數** 低於閾值（畫面偏糊）時，可 **強制 DFA 為 50%:50%**（`force_equal_dfa`），避免過度依賴不穩定的高頻，見 `core/model.py` 與 `capture/screen_detector.py`。

---

## 3. 訓練流程與資料管線

### 3.1 零儲存影片讀取

- **`core/dataloader.py`**：`VideoFramePairDataset`（別名 `DeepfakeDataset`）依影片路徑動態讀取影格，不預先落地大量圖檔。
- **標籤**：使用 `--real_dir` / `--fake_dir` 時由目錄強制標註；否則依路徑字串推斷（含 `original` / `real` 為 0，其餘為 1）。

### 3.2 增強（與訓練一致的重點）

- 壓縮／雜訊／JPEG／遮擋等（`augment_frame`）。
- **幾何／色彩**（僅訓練）：水平翻轉、±10° 旋轉、Color Jitter、高斯模糊；**FFT 在幾何變換之後重新計算**，與空間影像對齊。

### 3.3 訓練腳本 `core/train.py`

- **優化器**：AdamW（`weight_decay` 預設 `1e-4`）。
- **學習率**：`ReduceLROnPlateau` 監控 **val_loss**，`patience` 預設 2，`factor` 0.5。
- **正則**：Label smoothing（可選）、Dropout、**DFA 頻域暖機**（前若干 epoch 若頻域權重過低則加懲罰項，可關閉）。
- **輸出**：`weights/specxnet_best.pth`（依 **val_acc** 最佳）、`specxnet_latest.pth`；並寫入 **ROC、混淆矩陣、DFA 權重分佈圖、metrics.json**。

---

## 4. 評估指標與分析

### 4.1 訓練後自動產出

- **ROC / AUC**：自實作二元 ROC 與梯形積分近似 AUC。
- **混淆矩陣**、**準確率** 等寫入 `metrics.json`。
- **DFA 權重**：空間／頻域 softmax 權重的直方圖統計。

### 4.2 離線 rPPG 與統計（見下節）

- 由 `signal_analytics.py` 產生每支影片的 BPM、SNR、頻譜重心等；`stat_reporter.py` 可做 **Mann-Whitney U**、小提琴圖等。

---

## 5. 即時偵測器 `capture/screen_detector.py`

### 5.1 執行方式

在專案根目錄、啟用 venv 後：

```bash
python capture/screen_detector.py
python capture/screen_detector.py --weights weights/specxnet_best.pth
python capture/screen_detector.py --target_window "YouTube"
```

權重優先順序：若未指定 `--weights`，會嘗試 `weights/specxnet_best.pth` 等。無權重時仍會使用 **ImageNet 預訓練骨干**，但分類頭未訓練，**Fake 分數僅供測試管線**。

### 5.2 管線摘要

1. **螢幕擷取**（`mss`），可選 `--target_window` 對應視窗區域。
2. **人臉**：優先 **MediaPipe FaceDetector (tasks)**；失敗則 **畫面中央 ROI**。
3. **MediaPipe 框**：以中心 **放大 1.2 倍** 再裁切，模擬較寬鬆構圖。
4. **預處理**：與訓練一致的 **resize 策略** + **Unsharp Mask**；**Laplacian 變異數** 低於閾值時 **DFA 強制 50:50**。
5. **分數**：
   - **raw**：模型 sigmoid 輸出。
   - **校準**：\(\text{clip}((\text{raw}-0.3)/0.7,\,0,\,1)\)。
   - **平滑**：校準後分數進入 **長度 15** 的移動平均 → **avg**，UI 圓環以 **avg** 為主、**raw** 為輔。
6. **rPPG / BPM**：
   - 額頭／雙頰區域 RGB 均值時序 → **POS 類** 投影 → **Butterworth 帶通**。
   - **取樣率**：以連續影格到達時間的 **平均 \(dt\)** 得 \(f_s=1/\mathrm{mean}(dt)\)（有上下限），**不再寫死 30 Hz**。
   - **信賴度**：在 **1.1–1.3 Hz** 附近能量與 **0.5–4 Hz** 其餘頻帶能量比 **≥ 1.5** 時才視為 **BPM 可信**，否則 UI 顯示 **「偵測中」**。
7. **UI**：雙視窗（Live + Console）、Fake 圓環、FFT 縮圖、波形、FPS、延遲、穩定度；可寫入 `logs/session_*.csv`。

---

## 6. rPPG 研究管線

適用於 **批次影片** 與 **Real/Fake 對照分析**，與即時螢幕偵測邏輯分離但概念一致。

| 腳本 | 作用 |
|------|------|
| **`advanced_extractor.py`** | MediaPipe Face Mesh、ROI 加權 RGB、POS 訊號，輸出 CSV。 |
| **`signal_analytics.py`** | 讀取訊號、去趨勢、帶通、Welch、BPM/SNR/頻譜重心等。 |
| **`stat_reporter.py`** | 組間檢定、小提琴圖、報告 JSON/TXT。 |
| **`run_rppg_pipeline.py`** | 串接上述步驟。 |
| **`extract_rppg_signals.py`** | 轉呼叫 `advanced_extractor` 的入口。 |

---

## 7. 專案目錄結構（精簡）

```
PREVERVECT/
├── core/
│   ├── model.py          # SpecXNet + DFA
│   ├── dataloader.py     # 影片抽幀、增強
│   └── train.py          # 訓練與驗證指標
├── capture/
│   └── screen_detector.py
├── utils/
│   ├── fft_tools.py      # 功率譜
│   └── prepare_data.py   # 資料清理等
├── weights/              # 權重與訓練圖表（通常不提交 Git）
├── raw_data/             # 本機資料集（通常不提交 Git）
├── logs/                 # 偵測 session CSV
├── advanced_extractor.py
├── signal_analytics.py
├── stat_reporter.py
├── run_rppg_pipeline.py
├── requirements.txt
└── README.md
```

---

## 8. 環境與快速指令

### 8.1 安裝

```bash
python -m venv venv
# Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 8.2 訓練（範例）

```bash
python core/train.py ^
  --real_dir raw_data/original_sequences/youtube/c23/videos ^
  --fake_dir raw_data/manipulated_sequences/Deepfakes/c23/videos ^
  --save_dir weights ^
  --epochs 20 --batch_size 16 --num_workers 4
```

### 8.3 即時偵測

```bash
python capture/screen_detector.py --weights weights/specxnet_best.pth
```

### 8.4 依賴重點

- **PyTorch / torchvision / timm**：模型與訓練。
- **OpenCV**：影像與 UI。
- **MediaPipe**：臉部偵測（tasks）。
- **mss**：螢幕擷取。
- **SciPy**：濾波、Welch。
- **tqdm / matplotlib**：訓練進度與圖表。

---

## 版本說明

本 README 描述的是目前儲存庫中的 **設計與實作要點**；實際數值（閾值、epoch、資料路徑）可能因實驗而調整，請以程式碼與 `argparse` 說明為準。若在另一台電腦僅需閱讀架構，將本檔與 `core/`、`capture/` 一併複製即可對照；完整重現需 **Python 環境、權重檔與（若訓練）資料集**。
