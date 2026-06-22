# DurianAI Training Plan — KnockNet-lite + Vision CNN

> 从启发式规则到真正 AI 模型的完整路线图

---

## 📊 1. 数据资源现状（已确认）

### A. 声学（敲击音）数据集

| # | 数据集 | 样本数 | 分类体系 | 授权 | 格式 | 下载方式 | 质量 |
|---|--------|--------|---------|------|------|---------|------|
| 1 | **Zenodo 多模态** | 189 榴槤 | 3类 (Unripe 63/Ripe 63/Overripe 63) | CC BY | WAV (1.5 GB) | ✅ API自动 | ⭐⭐⭐⭐⭐ 完美均衡 |
| 2 | **Dalvii GitHub** | 100 WAV | 2类 (75-85% → unripe / 95%-Ripe → ripe) | 开源 | WAV | ✅ API自动 | ⭐⭐⭐⭐ 缺 overripe |
| 3 | KnockNet 论文 | 52 Monthong | 3类 (unripe/ripe/overripe) | 未公开 | 未明 | ❌ 需联系作者 | ⭐⭐⭐⭐⭐ 学术基准 |

**关键发现：Zenodo CSV 已下载分析**

CSV 文件 `durian_characteristics_cleaned.csv` 包含 189 行完整元数据：
- **Maturity_Stage**: Immature(63) / Mature(63) / Overmature(63)
- **Ripeness**: Unripe(63) / Ripe(63) / Overripe(63) — 完美均衡！
- **Actual_Ripening_Status**: Unripe(63) / Ripe(63) / Overripe(59) / Disease(2) / 空(2)
- **Code 格式**: `IM_CA_UN_1` = {Maturity}_{Class}_{Ripeness}_{Replicate}
  - IM/M/OM = Immature/Mature/Overmature
  - A/B/C = Class (批次)
  - UN/RI/OR = Unripe/Ripe/Overripe
- 传感器数据: 声学敲击音(1.5GB) + RGB(19GB) + 多光谱(928MB) + 热成像(512MB)

### B. 视觉（照片）数据集

| # | 数据集 | 图像数 | 分类体系 | 授权 | 格式 | 下载方式 | 质量 |
|---|--------|--------|---------|------|------|---------|------|
| 1 | **Roboflow xtned** | 1,438 张 | 3类 (Ripe/Unripe/Defect) | CC BY | JPG | ✅ 需 API Key | ⭐⭐⭐⭐⭐ 最佳起點 |
| 2 | **Rom1420 GitHub** | ? | 4类 (Ripe1=unripe/Ripe2=ripe/Ripe3=overripe/Ripe4=overripe) | 开源 | JPG | ✅ 免費 | ⭐⭐⭐⭐ CNN 93% |
| 3 | **Zenodo RGB** | 189 榴槤 × 多角度 | 3类 (同聲學) | CC BY | JPG (19GB) | ✅ API自動 | ⭐⭐⭐⭐⭐ 配對聲學 |
| 4 | Monthong 論文 | 1,000 | 4类 (overripe/semi/unripe/ripe) | 未公開 | 未明 | ❌ | MobileNetV2 95.5% |

---

## 🏷️ 2. 统一标签映射体系

**核心决策：统一采用 3 类体系**

| 统一标签 | Zenodo | Dalvii | Roboflow | Rom1420 | 含义 |
|---------|--------|--------|----------|---------|------|
| **unripe** | Unripe (UN) | 75-85% | Unripe | Ripe1 | 未熟/不熟 |
| **ripe** | Ripe (RI) | 95%-Ripe | Ripe | Ripe2 | 刚好成熟 |
| **overripe** | Overripe (OR) | — | — | Ripe3+Ripe4 | 过熟 |

**特殊处理：**
- Dalvii 缺 overripe 类 → 仅用于 unripe/ripe 二分类辅助训练
- Roboflow 的 Defect 类 → 排除出 3 分类（Phase 2 可加回为第 4 类）
- Zenodo 的 Disease 样本(2个) → 排除出训练集
- Zenodo CSV 的 Actual_Ripening_Status 为空(2个) → 使用 Ripeness 字段作 fallback

---

## 🏗️ 3. 模型架构方案

### Phase 1: 基线模型（1-2 周）

#### 声学: KnockNet-lite CNN on Mel Spectrogram

```
Input: (40, 125, 1) = 40 mel bins × 125 time frames (2s @ 16kHz)

Block 1: Conv2D(32, 3×3) → BN → Conv2D(32, 3×3) → BN → MaxPool(2×2) → Dropout(0.25)
Block 2: Conv2D(64, 3×3) → BN → Conv2D(64, 3×3) → BN → MaxPool(2×2) → Dropout(0.25)
Block 3: Conv2D(128, 3×3) → BN → GlobalAvgPool → Dropout(0.4)
Head:    Dense(64, relu) → Dropout(0.3) → Dense(3, softmax)

Output: ~1MB (TFLite INT8)
Target: 80%+ accuracy
```

#### 视觉: MobileNetV2 Transfer Learning

```
Input: (224, 224, 3)

Stage 1 (frozen base):
  MobileNetV2(ImageNet) → GAP → Dense(256) → Dropout(0.4) → Dense(128) → Dropout(0.3) → Dense(3, softmax)
  LR: 1e-3, epochs: 50

Stage 2 (fine-tune top 30%):
  Unfreeze top 30% of MobileNetV2
  LR: 1e-4, epochs: 20

Output: ~4MB (TFLite INT8)
Target: 85%+ accuracy
```

### Phase 3: 高精度模型（1-2 月）

| 模型 | 架构 | 预期大小 | 预期准确率 |
|------|------|---------|-----------|
| KnockNet 双流 | ConvMixer(MFCC) + ConvMixer(MGDCC) + LCA 融合 | ~3MB | 93%+ |
| EfficientNet-Lite0 | ImageNet → 自定义分类头 | ~5MB | 93%+ |

---

## 🔧 4. 训练超参数配置

### 声学 KnockNet-lite

| 参数 | 值 | 说明 |
|------|---|------|
| 采样率 | 16kHz | 统一降采样 |
| 特征 | Mel Spectrogram (40 bins) | Phase 1 用 mel；Phase 3 加 MFCC+MGDCC |
| 窗口 | 2s, 50% overlap | 从长录音切出多段 |
| n_fft | 512 | 32ms 窗口 |
| hop_length | 256 | 16ms 步长 |
| epochs | 100 | EarlyStopping patience=15 |
| batch_size | 32 | |
| LR | 1e-3 | ReduceLROnPlateau factor=0.5 |
| 优化器 | Adam | |
| Loss | sparse_categorical_crossentropy | |
| 数据增强 | SpecAugment (f_mask=8, t_mask=16) × 3 | 频谱遮蔽 |
| 分割 | 70/15/15 stratified | |

### 视觉 MobileNetV2

| 参数 | 值 | 说明 |
|------|---|------|
| 图像尺寸 | 224×224 | MobileNetV2 标准 |
| Stage 1 epochs | 50 | 分类头训练 |
| Stage 2 epochs | 20 | 微调 top 30% |
| Stage 1 LR | 1e-3 | |
| Stage 2 LR | 1e-4 | 微调需更低 LR |
| batch_size | 32 | |
| 优化器 | Adam | |
| Loss | sparse_categorical_crossentropy | |
| 数据增强 | 翻转/旋转±15°/亮度×0.8-1.2/对比度×0.8-1.3/色彩/模糊 | |
| 分割 | 使用 Roboflow 预设 split | |

---

## 📈 5. 预期准确率路线图

| 阶段 | 时间 | 聲學 | 視覺 | 融合(0.6+0.4) | 模型大小 |
|------|------|------|------|-------------|---------|
| 当前（启发式） | 已完成 | ~55% | ~50% | ~55% | 0 |
| **Phase 1** | 1-2週 | **80%+** | **85%+** | **85-88%** | ~5MB |
| Phase 2 | 2-4週 | 88%+ | 90%+ | 90-93% | ~5MB |
| Phase 3 | 1-2月 | 93%+ | 93%+ | **94-97%** | ~8MB |

---

## 🔀 6. 数据处理流程

### 声学数据流

```
Zenodo WAV (189 samples)
  ↓ load_and_segment(2s, 50% overlap)
  ↓ 每个样本 → 多个 2s 段
  ↓ extract_mel_spectrogram(40 bins)
  ↓ CSV Code → label (UN→unripe, RI→ripe, OR→overripe)
  ↓ 排除 Disease(2) + 空(2)

Dalvii WAV (100 files)
  ↓ load_and_segment(2s, 50% overlap)
  ↓ filename → label (75-85%→unripe, 95%-Ripe→ripe)
  ↓ (仅有 unripe/ripe, 无 overripe)

合并 → SpecAugment × 3 → stratified 70/15/15 split
```

### 视觉数据流

```
Roboflow (1,438 images, 3-class)
  ↓ folder structure: train/{Ripe,Unripe,Defect}/
  ↓ resize 224×224 → normalize [0,1]
  ↓ Ripe→ripe, Unripe→unripe, Defect→排除(Phase 1)

Zenodo RGB (189 × 多角度, 19GB)
  ↓ CSV Code → label
  ↓ resize 224×224 → normalize

合并 → augmentation(翻转/亮度/对比度) → split
```

---

## 📋 7. 执行步骤清单

### Phase 1 立即行动（Google Colab T4 GPU）

| Step | 行动 | 预计时间 | 备注 |
|------|------|---------|------|
| 1 | 注册 Roboflow 免费帐号，获取 API Key | 5 min | https://roboflow.com |
| 2 | 打开 `03_train_acoustic.ipynb` 在 Colab 运行 | 30 min | 自动下载 Zenodo+Dalvii |
| 3 | 下载 `knocknet_lite.tflite` + `acoustic_labels.txt` | 1 min | |
| 4 | 打开 `04_train_vision.ipynb` 在 Colab 运行 | 30 min | 需 Roboflow API Key |
| 5 | 下载 `durian_cnn.tflite` + `vision_labels.txt` | 1 min | |
| 6 | 将 TFLite 模型放入 `frontend/public/models/` | 5 min | |
| 7 | 重建前端 + 测试浏览器端推理 | 10 min | |
| 8 | 在手机实测验证 | 10 min | |

### Phase 2 数据增强（后续迭代）

| Step | 行动 | 备注 |
|------|------|------|
| 1 | 增强音频: 时间拉伸/音高偏移/噪音混合 × 5-10 | `ml/audio/augment.py` |
| 2 | 增强视觉: CutOut/MixUp/不同光照 × 5 | `ml/vision/augment.py` |
| 3 | App 内添加「验证结果」功能 | 用户开果后回报 |
| 4 | 收集真实手機錄音/照片作為新訓練數據 | 用户反馈闭环 |
| 5 | 持续迭代模型 | 定期重新训练 |

### Phase 3 高精度（KnockNet 重现）

| Step | 行动 | 备注 |
|------|------|------|
| 1 | 联系 Phapatanaburi 获取 KnockNet 52 样本数据 | 作者邮箱需查论文 |
| 2 | 实现 MGDCC (Modified Group Delay Cepstral Coefficients) 提取 | Phase domain feature |
| 3 | 实现 KnockNet 双流 ConvMixer + LCA | 参考 KnockNet 论文 |
| 4 | 训练 KnockNet → 93%+ | |
| 5 | 视觉改用 EfficientNet-Lite0 → 93%+ | |

---

## 📊 8. 评估指标

| 指标 | 用途 | 目标 |
|------|------|------|
| **Overall Accuracy** | 主要指标 | Phase 1: 80%+ (声) / 85%+ (视) |
| **Per-class Accuracy** | 确保每类都好 | 每类 > 75% |
| **Confusion Matrix** | 看哪类容易混淆 | unripe/ripe 混淆要少 |
| **F1-score (macro)** | 类不均衡时的指标 | > 0.80 |
| **TFLite inference time** | 手机浏览器速度 | < 200ms per inference |
| **TFLite model size** | 下载大小 | < 5MB |
| **AUC-ROC** | 多分类置信度 | > 0.90 |

---

## 🎯 9. 部署策略

### 前端（浏览器端推理）

```typescript
// tf-helpers.ts: 已实现 fallback 机制
async classifyImage(imageElement) {
  // 1. 尝试加载 TFLite 模型
  // 2. 如果模型可用 → 真实 CNN 推理 → 返回 {label, confidence}
  // 3. 如果模型不可用 → 色彩启发式 → 返回估计结果
}
```

### 后端（FastAPI 声学推理）

```python
# audio_processor.py: 已实现 fallback 机制
def analyze_acoustic(audio_path):
  # 1. 尝试加载 TFLite 模型
  # 2. 如果模型可用 → Mel Spectrogram → CNN 推理 → 返回 {label, confidence}
  # 3. 如果模型不可用 → 频谱启发式 → 返回估计结果
```

### 融合算法（不变）

```typescript
// fusion.ts: 声学 ×0.6 + 视觉 ×0.4
function fuseResults(acoustic, vision) {
  weighted = {}
  for label in ['unripe', 'ripe', 'overripe']:
    weighted[label] = acoustic[label] * 0.6 + vision[label] * 0.4
  return { label: max(weighted), confidence: weighted[max(weighted)], breakdown: weighted }
}
```

---

## 🔬 10. 学术参考

1. **KnockNet**: Phapatanaburi et al. (2025). *A dual-stream CNN-based ConvMixer for durian ripeness classification using magnitude and phase features from knocking sounds*. Results in Engineering. DOI: 10.1016/j.rineng.2025.104216
2. **Monthong MobileNetV2**: Sukkasem et al. (2024). *Durian Ripeness Classification Using Deep Transfer Learning*. IC2IT 2024. 95.5% on 1000 Monthong images.
3. **Rom1420 CNN**: Rom1420 (2025). *Durian-Ripeness-Detection*. GitHub. 93% on cropped images.
4. **Zenodo Multi-Modal**: Mesa-Satina et al. (2026). *Multi-Modal Sensor Data for Durian Fruit Maturity Classification*. DOI: 10.5281/zenodo.18603795
5. **Dalvii**: Dalvii (2025). *durian-maturity-classification*. GitHub. 100 WAV Dona variety.
6. **YAMNet**: Google (2020). *YAMNet: Audio Event Classification*. TFHub.
7. **SpecAugment**: Park et al. (2019). *SpecAugment: A Simple Data Augmentation Method for ASR*. Interspeech.
