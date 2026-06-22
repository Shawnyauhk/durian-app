# DurianAI — 模型部署指南

## 概覽

系統使用雙模態 AI 推理：

| 模態 | 推理位置 | 格式 | 當前狀態 |
|------|---------|------|---------|
| 視覺（AI 眼）| 前端瀏覽器（TF.js）| `.json` + `.bin` shards | ⏳ 待部署 |
| 聲學（AI 耳）| 後端 FastAPI（TFLite）| `.tflite` | ⏳ 待部署 |

---

## Part 1：視覺模型（前端）

### 1.1 訓練後導出（Google Colab）

```python
# 在 04_train_vision.ipynb 的最後一步執行
import tensorflowjs as tfjs

# 導出為 TF.js 格式（前端用）
tfjs.converters.save_keras_model(model, 'tfjs_vision_model')

# 同時導出 TFLite（備用）
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open('vision_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

導出後會產生：
```
tfjs_vision_model/
├── model.json        # 模型架構
├── group1-shard1of1.bin   # 模型權重
```

### 1.2 部署到前端

**方法 A：直接放入 public 目錄（開發/本地）**
```
frontend/public/models/vision/
├── model.json
├── group1-shard1of1.bin   # 可能有多個 shard
├── labels.txt             # 已存在（無需替換）
└── metadata.json          # 可選
```

```json
// metadata.json 格式（可選）
{
  "version": "1.0.0",
  "created_at": "2026-06-22",
  "architecture": "MobileNetV2",
  "input_shape": [224, 224, 3],
  "classes": ["unripe", "ripe", "overripe"],
  "accuracy_val": 0.85,
  "training_samples": 285
}
```

**方法 B：部署到 Render（生產環境）**

Render 前端是靜態托管，把模型文件放入 `public/models/vision/` 後 `git push` 即自動部署。

> ⚠️ 注意：`group1-shard1of1.bin` 可能有 4-8MB，確保 Render 免費方案支持（靜態文件 < 100MB）。

### 1.3 前端自動偵測

`tf-helpers.ts` 的加載邏輯：
1. 先用 `HEAD` 請求檢查 `/models/vision/model.json` 是否存在
2. 存在 → 加載 TF.js 模型，進行 AI 推理
3. 不存在 → 自動使用色彩啟發式（不報錯）

**部署後立即生效，無需任何代碼修改。**

---

## Part 2：聲學模型（後端）

### 2.1 訓練後導出（Google Colab）

```python
# 在 03_train_acoustic.ipynb 的最後一步執行

# 導出 TFLite INT8
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen  # 已在 notebook 定義
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32
tflite_model = converter.convert()

with open('knocknet_lite.tflite', 'wb') as f:
    f.write(tflite_model)
print(f'Model size: {len(tflite_model) / 1024:.1f} KB')  # 目標 ~1MB
```

同時創建 metadata：
```python
import json
meta = {
    "version": "1.0.0",
    "created_at": "2026-06-22",
    "architecture": "KnockNet-lite CNN",
    "input_shape": [1, 40, 125, 1],
    "n_mels": 40,
    "segment_duration": 2.0,
    "target_sr": 16000,
    "classes": ["unripe", "ripe", "overripe"],
    "accuracy_val": 0.82,
    "training_samples": 250
}
with open('knocknet_lite_metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)
```

### 2.2 部署到後端

**方法 A：本地/Render 環境**

把文件上傳到 `backend/models/acoustic/`：
```
backend/models/acoustic/
├── knocknet_lite.tflite
├── knocknet_lite_metadata.json   # 可選，用於版本顯示
└── labels.txt                    # 已存在（無需替換）
```

**方法 B：使用環境變量指定路徑**
```bash
# 在 Render 環境變量設置
ACOUSTIC_MODEL_PATH=/opt/render/project/src/backend/models/acoustic/knocknet_lite.tflite
ACOUSTIC_LABELS_PATH=/opt/render/project/src/backend/models/acoustic/labels.txt
```

**方法 C：熱重載（無需重啟）**

模型文件就位後，調用 API 熱重載：
```bash
curl -X POST https://your-api.onrender.com/api/model/reload
```

或在前端 Admin 頁面（待實作）觸發。

### 2.3 後端自動偵測

`audio_processor.py` 的加載邏輯：
1. 優先從 `ACOUSTIC_MODEL_PATH` 環境變量讀取路徑
2. 沒有環境變量 → 使用 `backend/models/acoustic/knocknet_lite.tflite`
3. 文件存在 → 加載 TFLite Interpreter，進行 AI 推理
4. 文件不存在 → 使用頻譜啟發式（不報錯）

---

## Part 3：模型狀態查看

### API 端點
```bash
# 查看模型加載狀態
GET /api/model-status
# → { acoustic_model_loaded: true/false, acoustic_model_version: "1.0.0", ... }

# 查看整體健康
GET /api/health
# → { status: "ok", acoustic_model_loaded: true, acoustic_inference_method: "tflite_model", ... }

# 熱重載模型
POST /api/model/reload
```

### 前端 UI
- Header 右側顯示兩個 badge：`📸 AI/啟發式` 和 `🔊 AI/啟發式`
- 首頁顯示「AI 模型推理中」或「啟發式規則分析」卡片
- 結果頁顯示使用的模型類型

---

## Part 4：Render 生產環境部署

### 前端（靜態站點）
```yaml
# render.yaml
services:
  - type: web
    name: durianai-frontend
    env: static
    buildCommand: cd frontend && npm install && npm run build
    staticPublishPath: frontend/dist
    envVars:
      - key: VITE_API_URL
        value: https://durianai-api.onrender.com
```

### 後端（Web Service）
```yaml
  - type: web
    name: durianai-api
    env: python
    buildCommand: pip install -r backend/requirements.txt
    startCommand: cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: ACOUSTIC_MODEL_PATH
        value: /opt/render/project/src/backend/models/acoustic/knocknet_lite.tflite
```

---

## Part 5：完整部署 Checklist

### Phase 1 模型部署（訓練後第一次）

- [ ] 在 Google Colab 運行 `03_train_acoustic.ipynb`，下載 `knocknet_lite.tflite`
- [ ] 在 Google Colab 運行 `04_train_vision.ipynb`，下載 `tfjs_vision_model/`
- [ ] 把 `knocknet_lite.tflite` 放到 `backend/models/acoustic/`
- [ ] 把 `tfjs_vision_model/*` 放到 `frontend/public/models/vision/`
- [ ] `git add . && git commit -m "deploy: phase1 models" && git push`
- [ ] Render 自動重新部署（前後端各 ~2-3 分鐘）
- [ ] 訪問 `GET /api/model-status` 確認 `acoustic_model_loaded: true`
- [ ] 在手機上打開 App，Header 應顯示 `📸 AI 🔊 AI`

### 後續版本更新
- [ ] 在 Colab 重新訓練（加入新數據）
- [ ] 更新 metadata.json 中的 version 字段
- [ ] 替換模型文件並 push
- [ ] 或調用 `POST /api/model/reload` 熱重載（無需重新部署）

---

## Part 6：模型性能預期

| 版本 | 聲學準確率 | 視覺準確率 | 融合準確率 | 推理速度 |
|------|-----------|-----------|-----------|---------|
| 當前（啟發式）| ~55% | ~50% | ~55% | <10ms |
| Phase 1 | 80%+ | 85%+ | **85-88%** | <50ms |
| Phase 2 | 88%+ | 90%+ | 90-93% | <50ms |
| Phase 3（KnockNet）| 93%+ | 93%+ | **94-97%** | <100ms |

---

## 附錄：模型文件格式說明

### TF.js SavedModel 格式（視覺，前端）
```
model.json            # 架構 + 權重分塊引用
group1-shard1of1.bin  # 權重數據（可能多個 shard）
```

### TFLite 格式（聲學，後端）
```
knocknet_lite.tflite  # 單文件，包含架構+量化權重
```

### 標籤文件
```
labels.txt            # 每行一個類別，與模型輸出 index 對應
unripe                # index 0
ripe                  # index 1
overripe              # index 2
```
