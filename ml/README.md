# DurianAI ML Pipeline — 數據收集 + 模型訓練 + 部署

> 目標：將啟發式規則替換為真正訓練過的 AI 模型，達到論文水準（聲學 96%、視覺 93%+、融合 94-97%）

---

## 📊 可用數據資源全景

### A. 聲學（敲擊音）數據集

| # | 數據集 | 規模 | 格式 | 類別 | 授權 | 下載 | 質量 |
|---|--------|------|------|------|------|------|------|
| 1 | **Zenodo Multi-Modal Durian** (2026) | 189 榴槤樣本 + 聲學敲擊錄音 | WAV | 未公開（需查看README） | CC BY | ✅ 免費 | ⭐⭐⭐⭐⭐ 最完整 |
| 2 | **Dalvii/durian-maturity-classification** (GitHub) | 100 WAV 文件 | WAV | 2類 (75-85% / 95%-Ripe) | 開源 | ✅ 免費 | ⭐⭐⭐⭐ Dona品種 |
| 3 | **KnockNet 論文** (Phapatanaburi 2025) | 52 Monthong 榴槤 | 未公開 | 3類 (unripe/ripe/overripe) | 未公開 | ❌ 需聯繫作者 | ⭐⭐⭐⭐⭐ 學術基準 |
| 4 | **CMU 泰國論文** (2023) | 未明 | 未公開 | 多類 | 未公開 | ❌ 需聯繫作者 | ⭐⭐⭐ |
| 5 | **麒麟西瓜數據集** (IEEE DataPort) | 8.7GB，WAV+JPG | WAV | 甜度值 | 學術 | ❌ 需訂閱 | ⭐⭐⭐ 遷移學習用 |

### B. 視覺（照片）數據集

| # | 數據集 | 規模 | 格式 | 類別 | 授權 | 下載 | 質量 |
|---|--------|------|------|------|------|------|------|
| 1 | **Roboflow durian-ripeness-detection-xtned** | 1,438 張 | JPG | 3類 (Ripe/Unripe/Defect) | CC BY | ✅ 免費 | ⭐⭐⭐⭐ 最佳起點 |
| 2 | **Roboflow durian-ripeness (Towppys Nest)** | 未明 | JPG | 未明 | 未明 | ✅ 免費 | ⭐⭐⭐ |
| 3 | **Roboflow durian-xfcje** | 651 張 | JPG | durian_ripe | 未明 | ✅ 免費 | ⭐⭐⭐ |
| 4 | **Rom1420/Durian-Ripeness-Detection (GitHub)** | 4類 (Ripe1-4) + CNN | JPG | 4類成熟度 | 開源 | ✅ 免費 | ⭐⭐⭐⭐ CNN 93% |
| 5 | **Monthong 1000張論文** (Sukkasem 2024) | 1,000 張 | 未公開 | 4類 (overripe/semi/unripe/ripe) | 未公開 | ❌ | MobileNet-v2 95.5% |
| 6 | **Rustyrice/durian_classifier** (GitHub) | 品種分類 (D13/D24/D197) | JPG | 3類品種 | 開源 | ✅ 免費 | ⭐⭐⭐ 品種不是成熟度 |
| 7 | **Zenodo Multi-Modal Durian** (2026) | 189 樣本 + RGB 圖像 | 未明 | 未明 | CC BY | ✅ 免費 | ⭐⭐⭐⭐ 配對聲學數據 |

---

## 🎯 推薦方案：三階段遞進

### Phase 1：快速啟動（1-2 週）— 用現成數據集訓練基線模型

**聲學 KnockNet-lite：**
- 數據：Zenodo 189 樣本聲學數據 + Dalvii 100 個 WAV
- 架構：基於 YAMNet 遷移學習 → TFLite
- 訓練：Google Colab（免費 GPU）
- 目標：80%+ 準確率

**視覺 CNN：**
- 數據：Roboflow 1,438 張（Ripe/Unripe/Defect）
- 架構：MobileNetV2 遷移學習 → TFLite
- 訓練：Google Colab
- 目標：85%+ 準確率

### Phase 2：數據增強（2-4 週）— 擴充數據集 + 用戶反饋閉環

**聲學數據增強：**
- 時間拉伸（0.8x-1.2x）
- 音高偏移（-2 ~ +2 半音）
- 背景噪音混合（市場/街道環境音）
- 房間脈衝響應（RIR）模擬
- SpecAugment（頻譜遮蔽）
- 預期擴充 5-10 倍

**視覺數據增強：**
- 旋轉、翻轉、亮度/對比度
- CutOut / MixUp
- 不同光照條件模擬
- 從 YouTube/網頁爬取更多榴槤照片

**用戶反饋閉環：**
- App 中添加「驗證結果」功能
- 用戶開果後回報實際成熟度
- 收集真實手機錄音/照片作為新訓練數據
- 持續迭代模型

### Phase 3：高精度模型（1-2 月）— KnockNet 架構重現 + 精細調參

**聲學 KnockNet：**
- 重現 Phapatanaburi 雙流 ConvMixer 架構
- Magnitude (MFCC) + Phase (MGDCC) 雙流
- Local Channel Attention (LCA) 機制
- 目標：90%+ 準確率

**視覺 CNN：**
- 多品種支援（貓山王/黑刺/D24/D13）
- 4 類成熟度（unripe/semi-ripe/ripe/overripe）
- EfficientNet-Lite 或 MobileNetV3
- 目標：93%+ 準確率

---

## 🏗️ 模型架構選擇

### 聲學模型

| 方案 | 架構 | 大小 | 推理速度 | 瀏覽器部署 | 推薦 |
|------|------|------|---------|-----------|------|
| A. YAMNet 遷移 | MobileNetV1 + 自定義分類頭 | ~5MB | 快 | ✅ TFLite.js | Phase 1 |
| B. 自定義 CNN on MFCC | 4-conv CNN | ~1MB | 很快 | ✅ TF.js | Phase 1 備選 |
| C. KnockNet 雙流 | ConvMixer + LCA | ~3MB | 中 | ✅ TFLite.js | Phase 3 |

### 視覺模型

| 方案 | 架構 | 大小 | 推理速度 | 瀏覽器部署 | 推薦 |
|------|------|------|---------|-----------|------|
| A. MobileNetV2 遷移 | ImageNet 預訓 + 自定義分類頭 | ~4MB (INT8) | 快 | ✅ TFLite.js | Phase 1 |
| B. EfficientNet-Lite0 | 更高精度 | ~5MB (INT8) | 中 | ✅ TFLite.js | Phase 3 |
| C. MobileNetV3-Small | 最小最快 | ~2MB (INT8) | 很快 | ✅ TFLite.js | 備選 |

### 瀏覽器端推理方案

| 方式 | 優點 | 缺點 | 適用場景 |
|------|------|------|---------|
| **TFLite.js** | 官方支持、WASM 後端 | 無 WebGPU | 當前最佳 |
| **TF.js** | WebGPU/WebGL 加速 | 包體較大 | GPU 可用時 |
| **ONNX.js** | 靈活 | 社區較小 | 備選 |

---

## 📁 ML 目錄結構

```
ml/
├── README.md              # 本文件
├── data/
│   ├── raw/               # 原始下載數據
│   ├── processed/         # 預處理後數據
│   └── augmented/         # 增強後數據
├── audio/
│   ├── train_knocknet.py  # KnockNet-lite 訓練腳本
│   ├── dataset.py         # 聲學數據加載器
│   ├── augment.py         # 音頻數據增強
│   └── export_tflite.py   # 導出 TFLite
├── vision/
│   ├── train_cnn.py       # 視覺 CNN 訓練腳本
│   ├── dataset.py         # 圖像數據加載器
│   ├── augment.py         # 圖像數據增強
│   └── export_tflite.py   # 導出 TFLite
├── models/
│   ├── acoustic/
│   │   ├── knocknet_lite.tflite  # 最終模型
│   │   └── labels.txt
│   └── vision/
│       ├── durian_cnn.tflite     # 最終模型
│       └── labels.txt
├── scripts/
│   ├── download_zenodo.py        # 下載 Zenodo 數據集
│   ├── download_roboflow.py      # 下載 Roboflow 數據集
│   ├── download_dalvii.py        # 下載 Dalvii 音頻數據
│   ├── prepare_audio.py          # 音頻預處理管線
│   └── prepare_vision.py         # 圖像預處理管線
└── notebooks/
    ├── 01_eda_audio.ipynb        # 音頻探索性分析
    ├── 02_eda_vision.ipynb       # 圖像探索性分析
    ├── 03_train_acoustic.ipynb   # 聲學模型訓練
    └── 04_train_vision.ipynb     # 視覺模型訓練
```

---

## 🔧 訓練流程

### Step 1：下載數據

```bash
# Zenodo 多模態榴槤數據集（聲學 + RGB）
python ml/scripts/download_zenodo.py

# Roboflow 榴槤成熟度照片
python ml/scripts/download_roboflow.py

# Dalvii GitHub 音頻
python ml/scripts/download_dalvii.py
```

### Step 2：預處理

```bash
# 音頻：統一採樣率 16kHz → MFCC 特徵 → 標籤映射
python ml/scripts/prepare_audio.py

# 圖像：統一 224x224 → 歸一化 → 標籤映射
python ml/scripts/prepare_vision.py
```

### Step 3：訓練（Google Colab 推薦）

```bash
# 聲學模型
python ml/audio/train_knocknet.py --data ml/data/processed/audio/ --epochs 100 --batch 32

# 視覺模型
python ml/vision/train_cnn.py --data ml/data/processed/vision/ --epochs 50 --batch 32
```

### Step 4：導出 & 部署

```bash
# 導出 TFLite（INT8 量化）
python ml/audio/export_tflite.py --model ml/models/acoustic/best.h5 --output ml/models/acoustic/knocknet_lite.tflite
python ml/vision/export_tflite.py --model ml/models/vision/best.h5 --output ml/models/vision/durian_cnn.tflite

# 複製到前端 public 目錄
cp ml/models/acoustic/knocknet_lite.tflite frontend/public/models/
cp ml/models/vision/durian_cnn.tflite frontend/public/models/
```

---

## 📈 預期效果

| 階段 | 聲學準確率 | 視覺準確率 | 融合準確率 | 時間 |
|------|-----------|-----------|-----------|------|
| 當前（啟發式） | ~55% | ~50% | ~55% | 已完成 |
| Phase 1（基線模型） | 80%+ | 85%+ | 85-88% | 1-2 週 |
| Phase 2（數據增強） | 88%+ | 90%+ | 90-93% | 2-4 週 |
| Phase 3（高精度） | 93%+ | 93%+ | 94-97% | 1-2 月 |

---

## 🔬 學術參考

1. Phapatanaburi et al. (2025). *KnockNet: A Dual-Stream CNN-Based ConvMixer for Durian Ripeness Classification Using Magnitude and Phase Features from Knocking Sounds*. Results in Engineering. DOI: 10.1016/j.rineng.2025.104216
2. Sukkasem et al. (2024). *Durian Ripeness Classification Using Deep Transfer Learning*. IC2IT 2024. MobileNetV2 95.5% on 1000 Monthong images.
3. Rom1420 (2025). *Durian-Ripeness-Detection*. GitHub. YOLOv8 + Custom CNN, 93% on cropped images.
4. Dalvii (2025). *durian-maturity-classification*. GitHub. 100 WAV files, Dona variety.
5. Zenodo (2026). *Multi-Modal Sensor Data for Durian Fruit Maturity Classification*. 189 samples, CC BY. DOI: 10.5281/zenodo.18603795
