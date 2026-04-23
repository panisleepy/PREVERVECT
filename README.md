# PREVERVECT

**PREVERVECT** 是一套以 **SpecXNet** 為核心的 Deepfake 偵測研究系統：結合 **空間域（RGB）** 與 **頻域（FFT 功率譜）** 雙流特徵、**DFA（雙流融合）**，並可搭配 **rPPG（遠端心率）** 與即時 **螢幕擷取偵測器**。本文件說明整體架構、計算與分析方法、以及各模組如何執行。

---

## 目錄

1. [系統架構總覽 / 快速上手路徑](#1-系統架構總覽)
2. [核心模型：SpecXNet](#2-核心模型specxnet)
3. [訓練流程與資料管線](#3-訓練流程與資料管線)
4. [評估指標與分析](#4-評估指標與分析)
5. [即時偵測器 `capture/screen_detector.py`](#5-即時偵測器-capturescreen_detectorpy)
6. [rPPG 研究管線](#6-rppg-研究管線)
7. [專案目錄結構](#7-專案目錄結構精簡)
8. [取得專案與執行（含 venv）](#8-取得專案與執行含-venv)
9. [Chrome 外掛 + FastAPI 後端（瀏覽器管線）](#9-chrome-外掛--fastapi-後端瀏覽器管線)

---

### 對照表

| 任務 | 章節 |
|------------|----------|
| **桌面螢幕即時偵測**（OpenCV 視窗、`mss` 擷取） | [§5](#5-即時偵測器-capturescreen_detectorpy) 與 [§8.4](#84-即時偵測器在專案根目錄已-activate-venv) |
| **YouTube / 瀏覽器內影片 + 外掛浮動面板** | [§9](#9-chrome-外掛--fastapi-後端瀏覽器管線)（須先開 **FastAPI**，再載入 **Chrome 擴充功能**） |
| **環境與依賴** | [§8.2](#82-建立虛擬環境並安裝依賴) |
| **訓練與權重** | [§3](#3-訓練流程與資料管線)、[§8.3](#83-要不要訓練過的模型)、[§8.5](#85-訓練-specxnet需要-real--fake-影片目錄) |

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

在專案根目錄、啟用 venv 後（完整指令見 [§8](#8-取得專案與執行含-venv)）：

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
PREVERVECT/                    # 專案根目錄（以下指令皆假設已 cd 到此）
├── app.py                     # FastAPI：/detect、/health（Chrome 外掛呼叫）
├── setup_env.py               # 可選：下載 EfficientNet 參考權重、簡單環境自檢
├── core/
│   ├── model.py               # SpecXNet + DFA
│   ├── dataloader.py          # 影片抽幀、增強
│   └── train.py               # 訓練與驗證指標
├── capture/
│   └── screen_detector.py    # 桌面即時偵測器（螢幕擷取 + OpenCV）
├── extension/                 # Chrome 擴充功能（Manifest V3）
│   ├── manifest.json
│   ├── content.js             # 1 FPS 送圖至 FastAPI、更新浮動 UI
│   └── popup.html
├── utils/
│   ├── fft_tools.py           # 功率譜
│   └── prepare_data.py        # 抽幀／整理資料等
├── weights/                   # 訓練權重、BlazeFace .tflite（通常不完全提交 Git）
├── logs/                      # screen_detector session CSV（可選）
├── raw_data/                  # 本機資料集（若使用；通常不提交 Git）
├── advanced_extractor.py
├── signal_analytics.py
├── stat_reporter.py
├── run_rppg_pipeline.py
├── requirements.txt
└── README.md
```

---

## 8. 取得專案與執行（含 venv）

### 8.1 取得程式碼

**方式 A：Git（推薦）**

將 `YOUR_REPO_URL` 換成你的 GitHub 儲存庫網址。

```bash
git clone https://github.com/YOUR_USER/PREVERVECT.git
cd PREVERVECT
```

**方式 B：ZIP**

在 GitHub 網頁 **Code → Download ZIP**，解壓後進入資料夾（內容應含 `core/`、`capture/`、`requirements.txt`）。

> **注意**：`.gitignore` 通常會排除 `weights/*.pth`、`raw_data/`。新環境上不會自動有 **訓練好的權重**；請自行從雲端硬碟／隨身碟複製 `specxnet_best.pth` 到專案下的 `weights/`（見 §8.3）。

---

### 8.2 建立虛擬環境並安裝依賴

**Windows（PowerShell）**（請先 `cd` 到專案根目錄 `PREVERVECT`）：

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

若執行政策擋住啟用腳本，可先執行：`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`。

**Linux / macOS（bash）**：

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

之後每次開新終端機要先 **啟用同一個 venv**，再執行下面各節的 `python ...` 與 `uvicorn ...` 指令。

**（可選）首次下載參考權重與自檢**：

```powershell
python setup_env.py
```

會嘗試將 `efficientnet_b0` 預訓練狀態存到 `weights/`，並簡單測試 `mediapipe` / `mss`。與 **SpecXNet 微調權重**（`specxnet_best.pth`）無關，後者請自行複製或訓練產生。

---

### 8.3 要不要「訓練過的模型」？

| 你想做的事 | 是否需要 `specxnet_best.pth`（或自訓權重）？ |
|------------|-------------------------------------------|
| **只跑 UI、看擷取／人臉框／rPPG 波形、測流程** | **不需要**。未提供權重時會用 ImageNet 預訓練骨干 + 未訓練的頭，**Fake 分數不可靠**。 |
| **認真做 Deepfake 分數（Fake Score）** | **需要**。將權重放在 `weights/specxnet_best.pth`，或用 `--weights` 指到完整路徑。 |
| **自己訓練新權重** | 需要 **影片資料集** + 執行 `core/train.py`（§8.5），產出新的 `.pth`。 |

---

### 8.4 即時偵測器（在專案根目錄、已 `activate` venv）

**桌面版**指 `capture/screen_detector.py`：用 `mss` 擷取螢幕，OpenCV 顯示 **Live Feed** 與 **Console** 兩個視窗。

**不帶自訓權重（僅測管線）**：

```powershell
python capture\screen_detector.py
```

**使用訓練權重（建議）**：

```powershell
python capture\screen_detector.py --weights weights\specxnet_best.pth
```

**只擷取標題含某關鍵字的視窗（例如瀏覽器）**：

```powershell
python capture\screen_detector.py --weights weights\specxnet_best.pth --target_window "Chrome"
```

結束：在 OpenCV 視窗按 **`q`**。

> 註：此路線與 [§9](#9-chrome-外掛--fastapi-後端瀏覽器管線) 的 Chrome 外掛 **互不影響**；Chrome 版需要另開 FastAPI。

---

### 8.5 訓練 SpecXNet（需要 Real / Fake 影片目錄）

先準備好資料路徑（範例為 FaceForensics++ 風格），在專案根目錄執行：

**PowerShell：**

```powershell
python core\train.py `
  --real_dir "raw_data\original_sequences\youtube\c23\videos" `
  --fake_dir "raw_data\manipulated_sequences\Deepfakes\c23\videos" `
  --save_dir "weights" `
  --epochs 20 `
  --batch_size 16 `
  --num_workers 4
```

**bash：**

```bash
python core/train.py \
  --real_dir raw_data/original_sequences/youtube/c23/videos \
  --fake_dir raw_data/manipulated_sequences/Deepfakes/c23/videos \
  --save_dir weights \
  --epochs 20 --batch_size 16 --num_workers 4
```

訓練完成後使用 **`weights/specxnet_best.pth`**（驗證集最佳），再依 §8.4 跑偵測器。

---

### 8.6 離線 rPPG 管線（可選，需影片與路徑自調）

```powershell
python run_rppg_pipeline.py
```

實際參數請依 `run_rppg_pipeline.py`、`advanced_extractor.py` 內的 `argparse` 為準。

---

### 8.7 依賴重點

- **PyTorch / torchvision / timm**：模型與訓練。
- **OpenCV**：影像與 UI。
- **MediaPipe**：臉部偵測（tasks）。
- **mss**：螢幕擷取。
- **SciPy**：濾波、Welch。
- **tqdm / matplotlib**：訓練進度與圖表。

---

## 9. Chrome 外掛 + FastAPI 後端（瀏覽器管線）

此路徑與 **桌面版** `capture/screen_detector.py` **互相獨立**：兩者共用 **`core/model.py`** 與 **`weights/`** 內的 SpecXNet 權重，但 **請勿** 把 `app.py` 與螢幕偵測器當成同一支程式。

### 9.1 前置條件

1. 完成 [§8.1](#81-取得程式碼)（clone 或 ZIP）與 [§8.2](#82-建立虛擬環境並安裝依賴)（venv + `pip install -r requirements.txt`）。
2. 將 **`weights/specxnet_best.pth`**（或你訓練好的 `.pth`）放到專案根目錄下的 **`weights/`**（見 [§8.3](#83-要不要訓練過的模型)）。
3. 之後所有指令都在 **專案根目錄** 執行，且終端機已 **`activate` 同一個 venv**。

### 9.2 啟動後端（FastAPI）

在專案根目錄開啟終端機（venv 已啟用），執行：

**Windows（PowerShell）**

```powershell
cd "C:\Users\<你的使用者>\OneDrive\文件\PREVERVECT"
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

若終端機顯示 **`uvicorn` 無法辨識**，請一律改用 **`python -m uvicorn ...`**（不依賴 PATH）。

**Linux / macOS**

```bash
cd /path/to/PREVERVECT
source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

- 看到類似 `Uvicorn running on http://0.0.0.0:8000` 即表示成功。
- **`0.0.0.0`** 代表本機網卡皆可連線；瀏覽器與外掛請使用 **`http://127.0.0.1:8000`**（與 `extension/content.js` 內設定一致）。

**快速自檢**

```powershell
curl http://127.0.0.1:8000/health
```

預期回傳 JSON：`{"status":"ok"}`。

**第一次執行 `/detect` 時**：後端可能自動下載 MediaPipe **BlazeFace** 模型至 `weights/blaze_face_short_range.tflite`，需網路；請勿關閉防火牆阻擋 Python。

**Windows + 中文路徑提醒**：若 MediaPipe 報 `Unable to open file ... errno=-1`，後端會將模型複製到 **`C:\temp\prevervect_models\`**（ASCII 路徑）再載入；請確保該資料夾可寫入。

### 9.3 安裝 Chrome 擴充功能（前端）

1. 確認 **`extension/`** 資料夾內有 `manifest.json`、`content.js`、`popup.html`。
2. 開啟 Chrome，網址列輸入 **`chrome://extensions/`**。
3. 開啟右上 **「開發人員模式」**。
4. 點 **「載入未封裝項目」**，選取專案內的 **`extension`** 資料夾（不是整個 repo 根目錄，而是內含 `manifest.json` 的那層）。
5. 若更新過 `content.js` / `manifest.json`，請在外掛卡片上點 **重新載入**。

### 9.4 端到端操作流程（YouTube 範例）

1. 先保持 **§9.2** 的 **`python -m uvicorn ...` 終端視窗開著**（後端常駐）。
2. 開啟 **YouTube**（或其它有 `<video>` 的頁面）並 **播放影片**。
3. 頁面右上角應出現 **PREVERVECT** 浮動面板；約 **1 FPS** 會向本機 `POST /detect`。
4. **無臉**時面板會提示「未偵測到人臉」；有臉時會顯示 **Fake 分數** 與綠／黃／紅進度條（語意見 `content.js` 註解）。
5. **「中斷偵測 (Stop)」** 可停止送圖；若要再測，**重新整理頁面** 或再開一次該分頁（外掛會重新掛載）。

### 9.5 快速對照：桌面 detector vs Chrome 外掛

| 項目 | 桌面（`capture/screen_detector.py`） | Chrome（`extension/` + `app.py`） |
|------|-------------------------------------|----------------------------------|
| 開啟 | §8.4：`python capture\screen_detector.py ...` | §9.2：`python -m uvicorn app:app ...`，再 §9.3 載入外掛 |
| 擷取來源 | 螢幕 `mss` | 網頁 `<video>` 1 FPS canvas |
| UI | OpenCV **Live Feed** / **Console** | 網頁右上角浮動面板 |
| 權重 | `weights/specxnet_best.pth`（或 `--weights`） | 同上（後端讀取 `weights/`） |
| 關閉 | OpenCV 視窗按 **q** | Stop 按鈕或關分頁；後端用 Ctrl+C |
| 同時執行 | 可與 Chrome 管線並存，但會 **各載入一份模型**，較吃記憶體 | — |

### 9.6 API 約定（給自訂前端或除錯）

| 項目 | 內容 |
|------|------|
| 健康檢查 | `GET http://127.0.0.1:8000/health` |
| 偵測 | `POST http://127.0.0.1:8000/detect`，Body：`{"image_base64":"<Data URL 或純 Base64>"}` |
| 成功（有臉推論） | `{"fake_score": 0~1, "is_reliable": true/false}` |
| 無臉 | `{"fake_score": 0, "is_reliable": false, "message": "No face detected"}` |

後端會 **裁臉**（MediaPipe 優先、Haar 備援）、**擴框 20%**、正方形 ROI，再送 **SpecXNet**；與桌面版前處理細節不完全相同，分數可能略有差異。

### 9.7 常見問題

| 現象 | 建議 |
|------|------|
| 狀態顯示 **伺服器連線失敗** | 確認 `uvicorn` 是否在跑、埠號是否 **8000**、防火牆是否擋；關閉後再開請 **重新載入擴充功能** 並重整頁面。 |
| 狀態顯示 **後端錯誤 HTTP 500** | 看執行 `uvicorn` 的終端機 **錯誤堆疊**；多數與依賴、GPU、或暫存圖異常有關。 |
| Fake 分數不穩或「不準」 | 確認已使用 **自訓** `specxnet_best.pth`；YouTube 壓縮與訓練集分佈不同會造成 **domain gap**。 |
| PowerShell 說 **`uvicorn` 無法辨識** | 確認已 `Activate` venv；或使用 **`python -m uvicorn app:app ...`**（見 §9.2）。 |

---

## 版本說明

本 README 描述的是目前儲存庫中的 **設計與實作要點**；實際數值（閾值、epoch、資料路徑）可能因實驗而調整，請以程式碼與 `argparse` 說明為準。**clone / 下載 ZIP** 後，務必 **建立 venv、安裝 `requirements.txt`**，並依需求 **自行放入 `weights/specxnet_best.pth`** 或執行訓練。若使用 **Chrome 管線**，另需依 **§9** 啟動 **FastAPI** 並 **載入擴充功能**。
