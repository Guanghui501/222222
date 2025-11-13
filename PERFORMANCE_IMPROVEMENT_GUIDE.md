# 性能改进指南 - 接近论文水平

## 📊 当前性能差距

| 模型 | Test MAE | 与论文差距 |
|------|----------|-----------|
| **论文结果** | **0.27** | - |
| 您的后期融合 (100 epochs) | 0.334 | +23.7% ❌ |
| 原模型 | 0.341 | +26.3% ❌ |

**目标**: 将 MAE 从 0.334 降低到接近 0.27

---

## 🔧 已修复的关键问题

### ✅ 修复1: 融合方式错误（已推送 - commit 33bc288）

**问题**：使用简单平均导致信息损失
```python
# 之前（错误）❌
h = (enhanced_graph + enhanced_text) / 2.0  # 64维，信息损失50%

# 现在（正确）✅
h = torch.cat([enhanced_graph, enhanced_text], dim=1)  # 128维，信息完整
```

**预期改进**: MAE 应该降低 **0.02-0.04**（约5-12%），目标 MAE ≈ 0.29-0.31

**立即测试**：
```bash
cd /public/home/ghzhang/crysmmnet-main/src

# 下载修复后的代码
curl -o models/alignn.py \
  https://raw.githubusercontent.com/Guanghui501/222222/claude/parse-functionality-011CV5SzJtxst49fCeaEgG5G/crysmmnet-main/src/models/alignn.py

# 重新训练 100 epochs
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property mbj_bandgap \
    --use_cross_modal True \
    --epochs 100 \
    --batch_size 64
```

---

## 🎯 进一步改进方案

### 方案2: 延长训练时间 ⭐⭐⭐

**问题**: 只训练了100 epochs，模型可能还未收敛

**论文可能使用**: 300-500 epochs

**实验**:
```bash
# 训练 300 epochs
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property mbj_bandgap \
    --use_cross_modal True \
    --epochs 300 \
    --batch_size 64
```

**预期改进**: MAE 可能再降低 **0.01-0.03**，目标 MAE ≈ 0.27-0.29

**建议**: 先用修复后的代码训练100 epochs看效果，如果有改善再训练300 epochs

---

### 方案3: 调整学习率和warmup ⭐⭐⭐

**当前设置**:
```bash
--learning_rate 0.001
--warmup_steps 2000
```

**优化设置**:
```bash
# 选项A: 降低学习率（更稳定）
--learning_rate 0.0005 \
--warmup_steps 3000

# 选项B: 增加warmup（更平滑）
--learning_rate 0.001 \
--warmup_steps 5000

# 选项C: 两者结合（推荐）
--learning_rate 0.0008 \
--warmup_steps 4000
```

**预期改进**: MAE 降低 **0.005-0.015**

---

### 方案4: 调整模型容量 ⭐⭐

**当前设置**:
```bash
--alignn_layers 4
--gcn_layers 4
--hidden_features 256
```

**增加容量**（如果内存允许）:
```bash
--alignn_layers 6 \
--gcn_layers 6 \
--hidden_features 512
```

**或减小容量**（防止过拟合）:
```bash
--alignn_layers 3 \
--gcn_layers 3 \
--hidden_features 256 \
--weight_decay 1e-4  # 增加正则化
```

**预期改进**: 需要实验确定，可能 ±0.01-0.02

---

### 方案5: 数据增强和预处理 ⭐⭐

**可能的改进**:

1. **归一化目标值**:
```python
# 检查数据范围
import pandas as pd
data = pd.read_csv('dataset/jarvis/mbj_bandgap/description.csv')
print(data['target'].describe())

# 如果范围很大，考虑标准化
mean = data['target'].mean()
std = data['target'].std()
```

2. **移除异常值**:
```python
# 移除超过3个标准差的样本
filtered_data = data[np.abs(data['target'] - mean) < 3 * std]
```

3. **平衡数据分布**:
- 如果数据分布不均匀，考虑分层采样

---

### 方案6: 集成学习 ⭐

**思路**: 训练多个模型，取平均

```bash
# 训练5个不同随机种子的模型
for seed in 123 456 789 101 202; do
    python train_with_cross_modal_attention.py \
        --dataset jarvis \
        --property mbj_bandgap \
        --use_cross_modal True \
        --epochs 300 \
        --random_seed $seed \
        --output_dir ./output/seed_$seed/
done

# 然后在测试时取5个模型的平均预测
```

**预期改进**: MAE 降低 **0.01-0.02**

---

### 方案7: 超参数搜索 ⭐⭐⭐

使用网格搜索或贝叶斯优化找最佳参数组合：

```python
# 关键超参数空间
hyperparams = {
    'learning_rate': [0.0005, 0.001, 0.002],
    'batch_size': [32, 64, 128],
    'cross_modal_num_heads': [2, 4, 8],
    'cross_modal_hidden_dim': [128, 256, 512],
    'weight_decay': [0, 1e-5, 1e-4],
}
```

**实现**:
```bash
# 创建搜索脚本
cat > hyperparameter_search.sh << 'EOF'
#!/bin/bash

for lr in 0.0005 0.001 0.002; do
    for bs in 32 64 128; do
        for heads in 2 4 8; do
            echo "Testing lr=$lr, bs=$bs, heads=$heads"
            python train_with_cross_modal_attention.py \
                --dataset jarvis \
                --property mbj_bandgap \
                --use_cross_modal True \
                --learning_rate $lr \
                --batch_size $bs \
                --cross_modal_num_heads $heads \
                --epochs 100 \
                --output_dir ./output/search_lr${lr}_bs${bs}_h${heads}/
        done
    done
done
EOF

chmod +x hyperparameter_search.sh
bash hyperparameter_search.sh
```

---

### 方案8: 使用中期融合 ⭐⭐

**测试中期融合是否有帮助**:
```bash
# 中期融合 + 晚期融合组合
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property mbj_bandgap \
    --use_cross_modal True \
    --use_middle_fusion True \
    --middle_fusion_layers "2" \
    --epochs 200 \
    --learning_rate 0.0008
```

---

## 📋 推荐的实验顺序

### 阶段1: 验证修复（1-2天）

1. **立即执行** - 测试拼接修复:
```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis --property mbj_bandgap \
    --use_cross_modal True \
    --epochs 100 --batch_size 64
```
**预期**: MAE ≈ 0.29-0.31（从0.334降低）

---

### 阶段2: 延长训练（2-3天）

2. **如果阶段1有改善** - 训练更多epochs:
```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis --property mbj_bandgap \
    --use_cross_modal True \
    --epochs 300 --batch_size 64 \
    --learning_rate 0.0008 --warmup_steps 4000
```
**预期**: MAE ≈ 0.27-0.29

---

### 阶段3: 超参数优化（3-5天）

3. **如果还有差距** - 超参数搜索:
```bash
# 运行网格搜索
bash hyperparameter_search.sh

# 分析结果
python analyze_search_results.py
```

---

### 阶段4: 高级技巧（可选）

4. **如果需要进一步提升**:
   - 集成学习（5个模型平均）
   - 中期融合优化
   - 数据预处理改进

---

## 🎯 目标 MAE 路线图

| 阶段 | 方法 | 预期 MAE | 累计改进 |
|------|------|----------|---------|
| 当前 | 原实现（平均） | 0.334 | - |
| 阶段1 | 修复拼接 | 0.30 | ↓10% |
| 阶段2 | 延长训练+调参 | 0.28 | ↓16% |
| 阶段3 | 超参数优化 | 0.27 | ↓19% |
| 目标 | **论文水平** | **0.27** | ✅ |

---

## 💻 快速测试脚本

```bash
# 保存为 quick_test.sh
#!/bin/bash

echo "========================================"
echo "测试修复后的模型性能"
echo "========================================"

# 下载最新代码
cd /public/home/ghzhang/crysmmnet-main/src
curl -o models/alignn.py \
  https://raw.githubusercontent.com/Guanghui501/222222/claude/parse-functionality-011CV5SzJtxst49fCeaEgG5G/crysmmnet-main/src/models/alignn.py

# 快速测试（50 epochs，验证修复有效）
echo "开始快速测试（50 epochs）..."
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property mbj_bandgap \
    --use_cross_modal True \
    --epochs 50 \
    --batch_size 64 \
    --output_dir ./output/fixed_test/

# 检查结果
echo ""
echo "========================================"
echo "测试完成！查看结果："
echo "========================================"
tail -10 ./output/fixed_test/mbj_bandgap/test_metrics.csv
```

使用方法：
```bash
chmod +x quick_test.sh
bash quick_test.sh
```

---

## 📊 性能监控

训练时密切关注：

### 1. 学习曲线
```bash
# 查看训练/验证损失
tail -20 output/mbj_bandgap/train_metrics.csv
tail -20 output/mbj_bandgap/val_metrics.csv
```

**正常情况**:
- 训练loss持续下降
- 验证loss下降后趋于平稳
- 两者差距不大（<20%）

**问题信号**:
- 验证loss上升 → 过拟合，增加正则化
- 训练loss不降 → 欠拟合，增加容量或降低学习率

### 2. MAE趋势
```bash
# 绘制MAE曲线
python -c "
import pandas as pd
import matplotlib.pyplot as plt

val = pd.read_csv('output/mbj_bandgap/val_metrics.csv')
plt.plot(val['epoch'], val['mae'])
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('Validation MAE over Epochs')
plt.grid(True)
plt.savefig('mae_curve.png')
print(f'Best MAE: {val[\"mae\"].min():.4f} at epoch {val[\"mae\"].idxmin()}')
"
```

---

## ⚠️ 常见陷阱

### 1. 数据泄露
确保训练/验证/测试集严格分离，不要在训练时看到测试数据。

### 2. 过度拟合测试集
不要根据测试集调参，使用验证集选择超参数。

### 3. 不同数据集版本
确保使用的数据集版本与论文一致。

### 4. 评估指标计算
确认MAE计算方式与论文相同（有些论文用eV，有些用其他单位）。

---

## 🎓 论文对比核查清单

请检查以下项是否与论文一致：

- [ ] 数据集版本和大小
- [ ] 训练/验证/测试集划分比例
- [ ] Batch size
- [ ] Learning rate 和 scheduler
- [ ] 训练 epochs 数
- [ ] 模型层数（ALIGNN/GCN）
- [ ] 隐藏层维度
- [ ] Dropout率
- [ ] 是否使用数据增强
- [ ] 评估指标的具体定义
- [ ] 是否使用集成学习

---

## 📞 需要帮助？

如果执行阶段1后：
- ✅ MAE 有明显改善（<0.31）→ 继续阶段2
- ❌ MAE 没有改善或变差 → 告诉我具体数值，我会进一步诊断

**立即开始**: 运行快速测试脚本，验证修复是否有效！

```bash
cd /public/home/ghzhang/crysmmnet-main/src

# 下载最新代码
curl -o models/alignn.py \
  https://raw.githubusercontent.com/Guanghui501/222222/claude/parse-functionality-011CV5SzJtxst49fCeaEgG5G/crysmmnet-main/src/models/alignn.py

# 立即测试
python train_with_cross_modal_attention.py \
    --dataset jarvis --property mbj_bandgap \
    --use_cross_modal True \
    --epochs 100 --batch_size 64
```

期待您的好消息！🚀
