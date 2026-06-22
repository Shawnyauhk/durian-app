# DurianAI — 榴槤成熟度智能檢測 App

一個 Mobile-first PWA，利用手機 **AI 眼（攝像頭）+ AI 耳（麥克風）** 雙模態分析榴槤成熟度。

## 技術架構

```
前端 (React + TypeScript + Vite + PWA)
  ├── 📸 AI 眼：手機攝像頭拍照 → TFLite CNN 模型推理（後備：色彩啟發式）
  ├── 🔊 AI 耳：麥克風錄製敲擊聲 → FastAPI 後端 TFLite 聲學推理（後備：頻譜啟發式）
  └── 🧠 AI 融合：聲學 ×0.6 + 視覺 ×0.4 加權判定

後端 (FastAPI + Python + librosa + TFLite)
  └── POST /api/analyze-acoustic → TFLite 模型 / MFCC 頻譜特徵分析

ML Pipeline (TensorFlow + TFLite)
  ├── 聲學：KnockNet-lite CNN on Mel Spectrogram → TFLite INT8
  ├── 視覺：MobileNetV2 Transfer Learning → TFLite INT8
  └── 數據：Zenodo + Roboflow + Dalvii → 預處理 → 增強 → 訓練 → 導出
```

## 準確率路線圖

| 階段 | 聲學 | 視覺 | 融合 | 狀態 |
|------|------|------|------|------|
| 當前（啟發式） | ~55% | ~50% | ~55% | ✅ 已完成 |
| Phase 1（基線模型） | 80%+ | 85%+ | 85-88% | 🔧 ML Pipeline 已建置 |
| Phase 2（數據增強） | 88%+ | 90%+ | 90-93% | 📋 待執行 |
| Phase 3（高精度） | 93%+ | 93%+ | 94-97% | 🎯 目標 |

> **Phase 1 論文基準**：KnockNet 96.34%（聲學）、MobileNetV2 95.5%（視覺 Monthong 1000 張）

## 快速啟動

### 前端

```bash
cd frontend
npm install
npm run dev          # 開發模式 (http://localhost:5173)
npm run build        # 生產構建
npm run preview      # 預覽生產版本
```

### 後端

```bash
cd backend
pip install -r requirements.txt
python main.py       # 啟動 API (http://localhost:8000)
```

### 環境變量

前端 `.env.local`：
```
VITE_API_URL=http://localhost:8000    # 本地開發
# VITE_API_URL=https://your-api.onrender.com    # 生產
```

## ML Pipeline 使用

```bash
cd durian-app

# 安裝 ML 依賴
pip install -r ml/requirements.txt

# 一鍵執行完整 Pipeline
python ml/run_pipeline.py download    # 下載數據集
python ml/run_pipeline.py prepare     # 預處理
python ml/run_pipeline.py train       # 訓練模型
python ml/run_pipeline.py export      # 導出 TFLite → 複製到前端

# 或全部執行
python ml/run_pipeline.py all
```

### 可用數據集

| 數據集 | 類型 | 規模 | 授權 | 下載 |
|--------|------|------|------|------|
| Zenodo Multi-Modal | 聲學+RGB | 189 樣本 | CC BY | ✅ 自動 |
| Dalvii GitHub | 聲學 WAV | 100 文件 | 開源 | ✅ 自動 |
| Roboflow | 圖像 JPG | 1,438 張 | CC BY | 🔑 需 API Key |

## 部署到 Render

### 前端（靜態網站）
- Build Command: `cd frontend && npm install && npm run build`
- Publish Directory: `frontend/dist`
- 費用: 免費

### 後端（Web Service）
- Runtime: Python 3.11
- Build Command: `pip install -r backend/requirements.txt`
- Start Command: `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
- 費用: 免費（有冷啟動）或 $7/月（無冷啟動）

## 手機使用說明

1. 在手機瀏覽器開啟 App URL
2. 點擊「**開始檢測**」
3. **Step 1 — AI 眼**：授權攝像頭 → 拍攝榴槤正面
4. **Step 2 — AI 耳**：授權麥克風 → 用指關節在中線位置敲擊 3 次
5. 查看**融合判定結果**

## 後續優化路徑

- [x] 建立 ML Pipeline 基礎設施（數據下載 + 預處理 + 訓練 + 導出）
- [x] 前端/後端支持 TFLite 模型推理（自動後備到啟發式）
- [ ] 下載數據集並訓練 Phase 1 基線模型
- [ ] 收集更多榴槤敲擊音數據（用戶反饋閉環）
- [ ] 訓練 KnockNet 雙流 ConvMixer（Phase 3）
- [ ] 多品種支援（貓山王、黑刺、D24、D13）
- [ ] 端側聲學推理（瀏覽器 TFLite.js 離線模式）

## 學術參考

1. Phapatanaburi et al. (2025). *A Dual-Stream CNN-Based ConvMixer for Durian Ripeness Classification Using Magnitude and Phase Features from Knocking Sounds*. Results in Engineering. DOI: 10.1016/j.rineng.2025.104216
2. Sukkasem et al. (2024). *Durian Ripeness Classification Using Deep Transfer Learning*. IC2IT 2024. MobileNetV2 95.5% on 1000 Monthong images.
3. Rom1420 (2025). *Durian-Ripeness-Detection*. GitHub. YOLOv8 + Custom CNN, 93% on cropped images.
4. Zenodo (2026). *Multi-Modal Sensor Data for Durian Fruit Maturity Classification*. 189 samples. DOI: 10.5281/zenodo.18603795
5. Dalvii (2025). *durian-maturity-classification*. GitHub. 100 WAV files, Dona variety.
