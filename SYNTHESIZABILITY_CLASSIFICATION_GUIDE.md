# 晶体可合成性二分类训练指南

## 📖 概述

本指南介绍如何使用CrysMMNet进行晶体可合成性二分类任务：
- **输入**: 晶体结构（CIF文件）+ 文本描述
- **输出**: 可合成（label=1）或 不可合成（label=0）

## 🎯 任务说明

### 二分类任务

| 类别 | 标签 | 含义 | 数据来源 |
|------|------|------|----------|
| **可合成** | 1 | 可以在实验室中合成的晶体 | `train-zheng/` 文件夹 |
| **不可合成** | 0 | 理论预测但难以合成的晶体 | `train-fu/` 文件夹 |

### 模型架构

```
晶体结构(CIF) ──► ALIGNN ──► 图特征
                              ↓
文本描述       ──► MatSciBERT ──► 文本特征
                              ↓
                    跨模态注意力 + 对比学习
                              ↓
                    2分类输出 [可合成, 不可合成]
```

---

## 🚀 快速开始

### Step 1: 准备数据

#### 1.1 确认数据结构

```bash
# 检查你的CIF文件夹
ls train-zheng/  # 可合成晶体CIF文件
ls train-fu/     # 不可合成晶体CIF文件
```

#### 1.2 运行数据准备脚本

```bash
python prepare_synthesizability_data.py \
    --positive_dir ./train-zheng \
    --negative_dir ./train-fu \
    --output_dir ./synthesizability_data \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1
```

**输出**:
```
================================================================================
晶体可合成性二分类数据准备
================================================================================

处理可合成晶体 (正样本, label=1):
  目录: ./train-zheng
  找到 500 个CIF文件
  ✅ 成功加载 485 个正样本

处理不可合成晶体 (负样本, label=0):
  目录: ./train-fu
  找到 500 个CIF文件
  ✅ 成功加载 492 个负样本

数据集统计:
  总样本数: 977
  可合成 (label=1): 485 (49.6%)
  不可合成 (label=0): 492 (50.4%)

数据集划分:
  训练集: 781 样本 (80%)
    - 可合成: 388
    - 不可合成: 393
  验证集: 98 样本 (10%)
    - 可合成: 48
    - 不可合成: 50
  测试集: 98 样本 (10%)
    - 可合成: 49
    - 不可合成: 49

✅ 数据准备完成！
保存数据到: ./synthesizability_data/description.csv
```

#### 1.3 生成的文件结构

```
synthesizability_data/
├── description.csv          # 数据标签文件
├── cif/                    # 所有CIF文件的副本
│   ├── synth_pos_001.cif
│   ├── synth_pos_002.cif
│   ├── synth_neg_001.cif
│   └── ...
└── dataset_info.txt        # 数据集统计信息
```

#### 1.4 description.csv 格式

```csv
id,composition,label,text,split
synth_pos_001,SrTiO3,1,"SrTiO3 crystal structure with cubic...",train
synth_neg_001,Li2O,0,"Li2O crystal structure with cubic...",train
...
```

---

### Step 2: 训练模型

#### 2.1 基础训练（推荐）

```bash
python train_synthesizability.py \
    --data_dir ./synthesizability_data \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 0.001
```

#### 2.2 完整配置训练

```bash
python train_synthesizability.py \
    --data_dir ./synthesizability_data \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 0.001 \
    --weight_decay 1e-5 \
    --alignn_layers 4 \
    --hidden_features 256 \
    --use_cross_modal True \
    --cross_modal_num_heads 8 \
    --use_contrastive True \
    --contrastive_weight 0.1 \
    --output_dir ./output_synthesizability/
```

#### 2.3 训练输出示例

```
================================================================================
晶体可合成性二分类训练
================================================================================

数据配置:
  数据目录: ./synthesizability_data

训练配置:
  批次大小: 32
  训练轮数: 100
  学习率: 0.001

模型配置:
  ALIGNN层数: 4
  隐藏层维度: 256
  ✅ 二分类任务 (output_features=2)

跨模态注意力:
  启用: True
  注意力头数: 8

对比学习:
  启用: True
  损失权重: 0.1

================================================================================

🎯 检测到分类任务，自动使用 CrossEntropyLoss
🎯 分类任务指标: Loss, Accuracy

🔥 对比学习已启用:
  - 损失权重: 0.1
  - 温度参数: 0.1

Epoch 1:
  Train_Acc: 0.6234
  Val_Acc: 0.6156
  Test_Acc: 0.6089

Epoch 2:
  Train_Acc: 0.7145
  Val_Acc: 0.7089
  Test_Acc: 0.7023

...

Epoch 100:
  Train_Acc: 0.9456
  Val_Acc: 0.8867
  Test_Acc: 0.8745

Best_accuracy 0.8867

============================================================
Classification Metrics on Test Set:
============================================================
Accuracy:  0.8745
Precision: 0.8923
Recall:    0.8556
F1 Score:  0.8735
AUC:       0.9234
============================================================

训练完成！
总用时: 2.34 小时
```

---

### Step 3: 查看结果

#### 3.1 预测结果文件

```bash
cat output_synthesizability/prediction_results_test_set.csv
```

**输出格式**:
```csv
id,target,predicted_class,prob_class_0,prob_class_1
synth_pos_001,1,1,0.123456,0.876544    ← 正确预测为可合成，87.7%概率
synth_neg_002,0,0,0.934521,0.065479    ← 正确预测为不可合成，93.5%概率
synth_pos_003,1,1,0.234567,0.765433    ← 正确预测为可合成
synth_neg_004,0,1,0.456789,0.543211    ← ❌ 误分类
```

**列说明**:
- `id`: 样本ID
- `target`: 真实标签（0=不可合成，1=可合成）
- `predicted_class`: 预测标签（0或1）
- `prob_class_0`: 不可合成的概率
- `prob_class_1`: 可合成的概率

#### 3.2 分析预测结果

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 加载结果
results = pd.read_csv('output_synthesizability/prediction_results_test_set.csv')

# 混淆矩阵
cm = confusion_matrix(results['target'], results['predicted_class'])
disp = ConfusionMatrixDisplay(cm, display_labels=['不可合成', '可合成'])
disp.plot()
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')

# 输出:
#                预测不可合成  预测可合成
# 实际不可合成        45          4
# 实际可合成          8          41

# 准确率 = (45+41) / 98 = 87.8%
# 精确率 = 41 / (41+4) = 91.1%  (预测为可合成的样本中，实际可合成的比例)
# 召回率 = 41 / (41+8) = 83.7%  (实际可合成的样本中，被正确预测的比例)
```

---

## 📊 评估指标说明

### 主要指标

| 指标 | 公式 | 含义 | 适用场景 |
|------|------|------|----------|
| **Accuracy** | (TP+TN) / Total | 整体准确率 | 类别平衡时 |
| **Precision** | TP / (TP+FP) | 查准率 | 关注假阳性 |
| **Recall** | TP / (TP+FN) | 查全率 | 关注假阴性 |
| **F1 Score** | 2×P×R / (P+R) | 精确率和召回率调和平均 | 综合评估 |
| **AUC** | ROC曲线下面积 | 分类能力 | 推荐作为主要指标 |

### 混淆矩阵解读

```
                   预测不可合成  预测可合成
实际不可合成(0)        TN           FP
实际可合成(1)          FN           TP

TN (True Negative):  正确预测为不可合成
FP (False Positive): 错误预测为可合成（假阳性）
FN (False Negative): 错误预测为不可合成（假阴性）
TP (True Positive):  正确预测为可合成
```

**实际意义**:
- **FP（假阳性）**: 预测可合成但实际不可合成 → 浪费实验资源
- **FN（假阴性）**: 预测不可合成但实际可合成 → 错失潜在材料

---

## 🔧 进阶配置

### 1. 处理类别不平衡

如果数据集中可合成和不可合成样本差异很大（如80%:20%），可以使用加权损失：

修改 `train.py` 第235行：
```python
# 计算类别权重
num_pos = sum(1 for d in train_data if d['target']==1)
num_neg = sum(1 for d in train_data if d['target']==0)
weight = torch.tensor([num_pos/num_neg, 1.0])

# 使用加权损失
criterion = nn.CrossEntropyLoss(weight=weight.to(device))
```

### 2. 调整决策阈值

默认使用0.5作为阈值，可以调整：

```python
# 在预测时
threshold = 0.4  # 降低阈值，增加召回率
predicted_class = (prob_class_1 > threshold).astype(int)
```

### 3. 超参数调优

推荐调优顺序：

1. **学习率** (最重要)
   ```bash
   --learning_rate 0.0001  # 小学习率
   --learning_rate 0.001   # 中等学习率（推荐）
   --learning_rate 0.01    # 大学习率
   ```

2. **批次大小**
   ```bash
   --batch_size 16   # 小批次（显存不足时）
   --batch_size 32   # 中等批次（推荐）
   --batch_size 64   # 大批次（加快训练）
   ```

3. **对比学习权重**
   ```bash
   --contrastive_weight 0.05  # 弱对比学习
   --contrastive_weight 0.1   # 中等（推荐）
   --contrastive_weight 0.2   # 强对比学习
   ```

---

## 🐛 常见问题

### Q1: 训练准确率很高但验证准确率低

**原因**: 过拟合

**解决方案**:
```bash
# 增加dropout
--cross_modal_dropout 0.2  # 默认0.1

# 减少模型复杂度
--alignn_layers 3  # 默认4
--hidden_features 128  # 默认256

# 添加权重衰减
--weight_decay 1e-4  # 默认1e-5
```

### Q2: 类别0的准确率高，类别1的准确率低

**原因**: 类别不平衡或数据分布问题

**解决方案**:
- 检查数据集中两个类别的比例
- 使用加权损失（见进阶配置）
- 收集更多类别1的数据

### Q3: 训练很慢

**解决方案**:
```bash
# 使用预处理数据（如果数据量大）
# 参考之前的预处理脚本

# 增大批次大小
--batch_size 64

# 减少workers（如果CPU瓶颈）
--num_workers 0
```

### Q4: 显存不足

**解决方案**:
```bash
# 减小批次大小
--batch_size 16

# 减小隐藏层维度
--hidden_features 128

# 减小注意力头数
--cross_modal_num_heads 4
```

---

## 📈 性能优化建议

### 1. 数据准备阶段

- ✅ 确保CIF文件格式正确
- ✅ 检查数据质量（删除异常样本）
- ✅ 保持类别平衡（50%:50%最佳）
- ✅ 划分数据集时确保类别比例一致

### 2. 训练阶段

- ✅ 使用跨模态注意力和对比学习
- ✅ 监控训练和验证准确率曲线
- ✅ 及时停止训练避免过拟合
- ✅ 保存最佳模型（基于验证集）

### 3. 评估阶段

- ✅ 查看混淆矩阵识别问题
- ✅ 分析误分类样本
- ✅ 使用AUC作为主要指标
- ✅ 检查不同阈值下的性能

---

## 📝 完整工作流示例

```bash
# Step 1: 准备数据
python prepare_synthesizability_data.py \
    --positive_dir ./train-zheng \
    --negative_dir ./train-fu \
    --output_dir ./synthesizability_data

# Step 2: 训练模型
python train_synthesizability.py \
    --data_dir ./synthesizability_data \
    --batch_size 32 \
    --epochs 100 \
    --use_cross_modal True \
    --use_contrastive True \
    --output_dir ./output_synthesizability/

# Step 3: 查看结果
head -20 ./output_synthesizability/prediction_results_test_set.csv

# Step 4: 分析结果（Python）
python -c "
import pandas as pd
from sklearn.metrics import classification_report

df = pd.read_csv('output_synthesizability/prediction_results_test_set.csv')
print(classification_report(df['target'], df['predicted_class'],
                           target_names=['不可合成', '可合成']))
"
```

---

## 🎓 进一步提升

### 1. 集成学习

训练多个模型并投票：

```bash
# 训练5个不同随机种子的模型
for seed in 123 456 789 101 202; do
    python train_synthesizability.py \
        --random_seed $seed \
        --output_dir ./output_seed_${seed}/
done

# 集成预测（Python脚本）
# 多数投票或概率平均
```

### 2. 添加物理特征

在模型输入中添加：
- 晶格常数
- 空间群
- 形成能（如果有）
- 元素周期表特征

### 3. 文本描述优化

生成更详细的文本描述：
- 化学键信息
- 配位环境
- 结构对称性
- 类似已知材料

---

## 📚 参考资料

- ALIGNN论文: https://www.nature.com/articles/s41524-021-00650-1
- MatSciBERT: https://github.com/M3RG-IITD/MatSciBERT
- 对比学习: SimCLR, MoCo等方法

需要帮助？请查看项目README或提Issue！
