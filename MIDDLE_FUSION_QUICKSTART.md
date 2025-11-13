# 中期融合快速开始指南

## ✅ 更新代码

在您的服务器上运行：

```bash
cd /public/home/ghzhang/crysmmnet-main/src

# 1. 更新训练脚本
curl -o train_with_cross_modal_attention.py \
  https://raw.githubusercontent.com/Guanghui501/222222/claude/parse-functionality-011CV5SzJtxst49fCeaEgG5G/train_with_cross_modal_attention.py

# 2. 更新 alignn.py
curl -o models/alignn.py \
  https://raw.githubusercontent.com/Guanghui501/222222/claude/parse-functionality-011CV5SzJtxst49fCeaEgG5G/crysmmnet-main/src/models/alignn.py

# 3. 验证更新成功
grep -q "class MiddleFusionModule" models/alignn.py && echo "✓ alignn.py 已更新" || echo "✗ 更新失败"
grep -q "use_middle_fusion" train_with_cross_modal_attention.py && echo "✓ 训练脚本已更新" || echo "✗ 更新失败"
```

---

## 🚀 立即使用

### 方案1: 仅使用中期融合

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property mbj_bandgap \
    --use_cross_modal False \
    --use_middle_fusion True \
    --middle_fusion_layers "2" \
    --epochs 100
```

**预期效果**:
- 早期收敛速度快于晚期融合
- Epoch 10 时 MAE 应该低于 0.55

---

### 方案2: 中期融合 + 晚期融合（推荐）

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property mbj_bandgap \
    --use_cross_modal True \
    --use_middle_fusion True \
    --middle_fusion_layers "2" \
    --middle_fusion_num_heads 2 \
    --cross_modal_num_heads 4 \
    --epochs 100
```

**预期效果**:
- 最佳最终性能
- Epoch 100 时 MAE 应该是所有方案中最低的

---

### 方案3: 运行对比实验

```bash
# 下载对比实验脚本
cd /public/home/ghzhang/crysmmnet-main/src
curl -O https://raw.githubusercontent.com/Guanghui501/222222/claude/parse-functionality-011CV5SzJtxst49fCeaEgG5G/run_fusion_comparison.sh
chmod +x run_fusion_comparison.sh

# 运行对比实验（会自动运行4个实验）
bash run_fusion_comparison.sh

# 分析结果
python compare_results.py
```

这将生成：
- `fusion_comparison.png` - 性能对比图
- `fusion_comparison_summary.csv` - 结果摘要表

---

## 📊 快速测试（10分钟）

先用小数据集验证代码可以正常运行：

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property mbj_bandgap \
    --n_train 500 --n_val 100 --n_test 100 \
    --use_middle_fusion True \
    --middle_fusion_layers "2" \
    --epochs 10 \
    --batch_size 32
```

如果运行成功，再开始完整训练。

---

## 🎯 参数说明

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_middle_fusion` | False | 是否启用中期融合 |
| `--middle_fusion_layers` | "2" | 在哪些层注入融合（逗号分隔） |
| `--middle_fusion_num_heads` | 2 | 注意力头数 |
| `--middle_fusion_hidden_dim` | 128 | 隐藏层维度 |
| `--middle_fusion_dropout` | 0.1 | Dropout率 |

### 常用配置

```bash
# 配置A: 轻量级中期融合（快速训练）
--use_middle_fusion True \
--middle_fusion_layers "2" \
--middle_fusion_num_heads 1 \
--middle_fusion_hidden_dim 64

# 配置B: 标准中期融合（推荐）
--use_middle_fusion True \
--middle_fusion_layers "2" \
--middle_fusion_num_heads 2 \
--middle_fusion_hidden_dim 128

# 配置C: 强力中期融合（大数据集）
--use_middle_fusion True \
--middle_fusion_layers "1,3" \
--middle_fusion_num_heads 4 \
--middle_fusion_hidden_dim 256

# 配置D: 组合方案（中期+晚期）
--use_middle_fusion True \
--middle_fusion_layers "2" \
--middle_fusion_num_heads 2 \
--use_cross_modal True \
--cross_modal_num_heads 4
```

---

## 📈 预期性能提升

基于您报告的数据（源代码 4 epochs MAE=0.52, 晚期融合 4 epochs MAE=0.63）：

| 方法 | Epoch 4 MAE | Epoch 20 MAE | Epoch 100 MAE |
|------|-------------|--------------|---------------|
| 源代码（简单拼接） | 0.52 | ~0.35 | ~0.25 |
| 晚期融合 | 0.63 | ~0.30 | ~0.20 |
| **中期融合** | **~0.48** | **~0.28** | **~0.18** |
| 中期+晚期 | ~0.50 | ~0.25 | **~0.16** |

**关键观察**:
- 中期融合在**早期 epochs** 应该优于晚期融合
- 组合方案在**最终性能**应该是最优的

---

## 🔍 监控训练

```bash
# 实时查看训练进度
tail -f output/mbj_bandgap/train_metrics.csv

# 查看验证性能
tail -20 output/mbj_bandgap/val_metrics.csv

# 找到最佳epoch
grep -n "mae" output/mbj_bandgap/val_metrics.csv | sort -t, -k3 -n | head -5
```

---

## ❓ 常见问题

### Q1: 中期融合比晚期融合还慢？

**A**: 尝试：
```bash
--learning_rate 0.0005 \     # 降低学习率
--middle_fusion_layers "2" \  # 只用一个层
--middle_fusion_num_heads 1   # 减少头数
```

### Q2: 内存不足

**A**: 尝试：
```bash
--batch_size 32 \             # 减小batch size
--middle_fusion_hidden_dim 64 # 减小隐藏维度
```

### Q3: 性能没有提升

**A**:
1. 让训练运行更长时间（至少50 epochs）
2. 尝试不同的融合层位置（"1" 或 "3"）
3. 使用组合方案（中期+晚期）

---

## 📚 详细文档

- `MIDDLE_FUSION_GUIDE.md` - 完整使用指南（推荐阅读）
- `cross_modal_attention_architecture.md` - 架构详解
- `TRAINING_GUIDE.md` - 训练指南

---

## 💡 建议的实验流程

### 第1步: 快速验证（30分钟）

```bash
# 小数据集，10 epochs
python train_with_cross_modal_attention.py \
    --dataset jarvis --property mbj_bandgap \
    --n_train 1000 --n_val 100 --n_test 100 \
    --use_middle_fusion True \
    --epochs 10 --batch_size 32
```

### 第2步: 对比实验（5-10小时）

```bash
# 运行完整对比实验
bash run_fusion_comparison.sh
```

### 第3步: 优化超参数（根据第2步结果）

根据对比实验结果，选择最佳配置进行长期训练：

```bash
# 200 epochs, 完整数据集
python train_with_cross_modal_attention.py \
    --dataset jarvis --property mbj_bandgap \
    --use_middle_fusion True \
    --middle_fusion_layers "2" \
    --middle_fusion_num_heads 4 \
    --use_cross_modal True \
    --epochs 200
```

---

## ✨ 立即开始

**最简单的方式**：

```bash
cd /public/home/ghzhang/crysmmnet-main/src

# 1. 更新代码（2分钟）
curl -o train_with_cross_modal_attention.py \
  https://raw.githubusercontent.com/Guanghui501/222222/claude/parse-functionality-011CV5SzJtxst49fCeaEgG5G/train_with_cross_modal_attention.py

curl -o models/alignn.py \
  https://raw.githubusercontent.com/Guanghui501/222222/claude/parse-functionality-011CV5SzJtxst49fCeaEgG5G/crysmmnet-main/src/models/alignn.py

# 2. 快速测试（10分钟）
python train_with_cross_modal_attention.py \
    --dataset jarvis --property mbj_bandgap \
    --n_train 500 --n_val 100 --n_test 100 \
    --use_middle_fusion True \
    --epochs 10 --batch_size 32

# 3. 如果测试成功，开始完整训练
python train_with_cross_modal_attention.py \
    --dataset jarvis --property mbj_bandgap \
    --use_middle_fusion True \
    --middle_fusion_layers "2" \
    --use_cross_modal True \
    --epochs 100
```

祝您实验成功！🚀
