# CrysMMNet 跨模态注意力 - 训练指南

完整的训练代码和使用说明。

## 📁 文件清单

```
.
├── train_with_cross_modal_attention.py  # 主训练脚本（完整功能）
├── train_examples.sh                     # Bash脚本示例集合
├── simple_training_example.py            # Python简单示例
├── TRAINING_GUIDE.md                     # 本文件
├── cross_modal_attention_usage.md        # 详细使用文档
└── cross_modal_attention_architecture.md # 架构说明
```

---

## 🚀 快速开始

### 方法1: 使用完整训练脚本（推荐）

```bash
# 1. 查看所有参数
python train_with_cross_modal_attention.py --help

# 2. JARVIS 形成能训练
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --epochs 1000 \
    --batch_size 64

# 3. Material Project 带隙训练
python train_with_cross_modal_attention.py \
    --dataset mp \
    --property band_gap \
    --n_train 60000 \
    --n_val 5000 \
    --n_test 4132 \
    --use_cross_modal True \
    --epochs 1000 \
    --batch_size 64
```

### 方法2: 使用示例脚本

```bash
# 给脚本添加执行权限
chmod +x train_examples.sh

# 运行特定示例（编辑脚本选择需要的部分）
./train_examples.sh
```

### 方法3: 使用原始训练脚本

```bash
# 进入src目录
cd crysmmnet-main/src

# 使用原有的训练脚本（会自动使用跨模态注意力）
python train_folder.py \
    --root_dir '../dataset/' \
    --dataset 'Jarvis' \
    --property 'fe' \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --epochs 1000 \
    --batch_size 64
```

---

## 📋 完整参数说明

### 数据集参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--root_dir` | str | `./crysmmnet-main/dataset/` | 数据集根目录 |
| `--dataset` | str | `jarvis` | 数据集名称: jarvis, mp, toy |
| `--property` | str | `formation_energy` | 预测的性质 |

### 数据划分参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--train_ratio` | float | `0.8` | 训练集比例 |
| `--val_ratio` | float | `0.1` | 验证集比例 |
| `--test_ratio` | float | `0.1` | 测试集比例 |
| `--n_train` | int | `None` | 训练样本数（覆盖ratio） |
| `--n_val` | int | `None` | 验证样本数 |
| `--n_test` | int | `None` | 测试样本数 |

### 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--batch_size` | int | `64` | 批次大小 |
| `--epochs` | int | `1000` | 训练轮数 |
| `--learning_rate` | float | `0.001` | 学习率 |
| `--weight_decay` | float | `1e-5` | 权重衰减 |
| `--warmup_steps` | int | `2000` | Warmup步数 |

### 模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--alignn_layers` | int | `4` | ALIGNN层数 |
| `--gcn_layers` | int | `4` | GCN层数 |
| `--hidden_features` | int | `256` | 隐藏层维度 |

### 跨模态注意力参数 ⭐

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_cross_modal` | bool | `True` | 启用跨模态注意力 |
| `--cross_modal_hidden_dim` | int | `256` | 注意力隐藏维度 |
| `--cross_modal_num_heads` | int | `4` | 注意力头数 (1/2/4/8) |
| `--cross_modal_dropout` | float | `0.1` | Dropout率 |

### 其他参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--output_dir` | str | `./output/` | 输出目录 |
| `--resume` | int | `0` | 从checkpoint恢复 (0/1) |
| `--random_seed` | int | `123` | 随机种子 |
| `--num_workers` | int | `0` | 数据加载线程数 |

---

## 📊 支持的数据集和性质

### JARVIS-DFT 数据集

| Property Flag | 完整名称 | 数据集大小 |
|--------------|---------|-----------|
| `formation_energy` / `fe` | Formation Energy (per atom) | ~55,000 |
| `total_energy` | Total Energy | ~55,000 |
| `opt_bandgap` | Optical Bandgap | ~55,000 |
| `mbj_bandgap` | MBJ Bandgap | ~24,000 |
| `bulk_modulus_kv` | Bulk Modulus | ~14,000 |
| `shear_modulus_gv` | Shear Modulus | ~14,000 |

**使用示例**:
```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy \
    --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
```

### Material Project 数据集

| Property Flag | 完整名称 | 数据集划分 |
|--------------|---------|-----------|
| `formation_energy` | Formation Energy | 60000 / 5000 / 4132 |
| `band_gap` | Band Gap | 60000 / 5000 / 4132 |
| `bulk` | Bulk Modulus | 4664 / 393 / 393 |
| `shear` | Shear Modulus | 4664 / 393 / 393 |

**使用示例**:
```bash
python train_with_cross_modal_attention.py \
    --dataset mp \
    --property formation_energy \
    --n_train 60000 --n_val 5000 --n_test 4132
```

---

## 🔬 实验场景

### 场景1: 标准训练

```bash
# 使用推荐的跨模态注意力配置
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --cross_modal_hidden_dim 256 \
    --cross_modal_dropout 0.1 \
    --epochs 1000 \
    --batch_size 64
```

### 场景2: 对比实验

```bash
# 实验A: Baseline（不使用跨模态注意力）
python train_with_cross_modal_attention.py \
    --dataset jarvis --property formation_energy \
    --use_cross_modal False \
    --output_dir ./output/baseline/

# 实验B: 跨模态注意力
python train_with_cross_modal_attention.py \
    --dataset jarvis --property formation_energy \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --output_dir ./output/cross_modal/
```

### 场景3: 消融实验（不同头数）

```bash
# 2个头
python train_with_cross_modal_attention.py \
    --dataset jarvis --property formation_energy \
    --cross_modal_num_heads 2 \
    --output_dir ./output/heads_2/

# 4个头（推荐）
python train_with_cross_modal_attention.py \
    --dataset jarvis --property formation_energy \
    --cross_modal_num_heads 4 \
    --output_dir ./output/heads_4/

# 8个头
python train_with_cross_modal_attention.py \
    --dataset jarvis --property formation_energy \
    --cross_modal_num_heads 8 \
    --output_dir ./output/heads_8/
```

### 场景4: 超参数调优

```bash
# 小模型（显存有限）
python train_with_cross_modal_attention.py \
    --dataset jarvis --property formation_energy \
    --hidden_features 128 \
    --cross_modal_hidden_dim 128 \
    --cross_modal_num_heads 2 \
    --batch_size 32

# 大模型（性能优先）
python train_with_cross_modal_attention.py \
    --dataset jarvis --property formation_energy \
    --hidden_features 512 \
    --cross_modal_hidden_dim 512 \
    --cross_modal_num_heads 8 \
    --batch_size 64
```

### 场景5: 快速测试

```bash
# 小数据集，少epoch，快速验证代码
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy \
    --n_train 1000 --n_val 100 --n_test 100 \
    --epochs 10 \
    --batch_size 32
```

### 场景6: 批量训练所有性质

```bash
# 使用Bash循环
for prop in formation_energy total_energy opt_bandgap mbj_bandgap bulk_modulus_kv shear_modulus_gv; do
    echo "训练性质: $prop"
    python train_with_cross_modal_attention.py \
        --dataset jarvis \
        --property $prop \
        --use_cross_modal True \
        --epochs 1000 \
        --batch_size 64 \
        --output_dir ./output/jarvis_${prop}/
done
```

---

## 📈 训练监控

### 输出文件结构

训练完成后，输出目录结构如下：

```
output/
└── formation_energy/
    ├── config.json                 # 训练配置
    ├── checkpoint_best.pt          # 最佳模型
    ├── checkpoint_100.pt           # 第100轮检查点
    ├── checkpoint_200.pt           # 第200轮检查点
    ├── train_predictions.csv       # 训练集预测结果
    ├── val_predictions.csv         # 验证集预测结果
    ├── test_predictions.csv        # 测试集预测结果
    └── history.json                # 训练历史
```

### 查看训练结果

```python
import json
import pandas as pd

# 读取训练历史
with open('output/formation_energy/history.json', 'r') as f:
    history = json.load(f)

print(f"最佳验证MAE: {min(history['val_mae']):.4f}")
print(f"最终测试MAE: {history['test_mae'][-1]:.4f}")

# 读取预测结果
test_pred = pd.read_csv('output/formation_energy/test_predictions.csv')
print(test_pred.head())
```

---

## 🐛 故障排除

### 问题1: CUDA out of memory

**解决方案**:
```bash
# 减小batch size
--batch_size 32

# 减小模型大小
--hidden_features 128 \
--cross_modal_hidden_dim 128 \
--cross_modal_num_heads 2
```

### 问题2: 数据集路径错误

**解决方案**:
```bash
# 确保数据集路径正确
ls ./crysmmnet-main/dataset/jarvis/formation_energy_peratom/

# 或指定完整路径
--root_dir /absolute/path/to/dataset/
```

### 问题3: 维度不匹配错误

**错误信息**: `hidden_dim must be divisible by num_heads`

**解决方案**:
```bash
# 确保 hidden_dim 能被 num_heads 整除
--cross_modal_hidden_dim 256 --cross_modal_num_heads 4  # ✓ 256/4=64
--cross_modal_hidden_dim 256 --cross_modal_num_heads 3  # ✗ 256/3=85.33
```

### 问题4: 训练不稳定

**解决方案**:
```bash
# 增加dropout
--cross_modal_dropout 0.2

# 减小学习率
--learning_rate 0.0005

# 增加warmup步数
--warmup_steps 5000
```

---

## 💡 最佳实践

### 1. 推荐配置

**标准场景**（推荐）:
```bash
python train_with_cross_modal_attention.py \
    --use_cross_modal True \
    --cross_modal_hidden_dim 256 \
    --cross_modal_num_heads 4 \
    --cross_modal_dropout 0.1 \
    --batch_size 64 \
    --learning_rate 0.001
```

**显存受限**:
```bash
python train_with_cross_modal_attention.py \
    --use_cross_modal True \
    --hidden_features 128 \
    --cross_modal_hidden_dim 128 \
    --cross_modal_num_heads 2 \
    --batch_size 32
```

**大数据集/高性能**:
```bash
python train_with_cross_modal_attention.py \
    --use_cross_modal True \
    --hidden_features 512 \
    --cross_modal_hidden_dim 512 \
    --cross_modal_num_heads 8 \
    --batch_size 64 \
    --learning_rate 0.0005
```

### 2. 超参数调优顺序

1. **首先**: 固定其他参数，调整 `cross_modal_num_heads` (2, 4, 8)
2. **然后**: 调整 `cross_modal_hidden_dim` (128, 256, 512)
3. **最后**: 微调 `cross_modal_dropout` (0.05, 0.1, 0.2)

### 3. 实验记录

建议使用以下格式记录实验：

```bash
# 实验日志
echo "Experiment: cross_modal_heads_4" >> experiments.log
echo "Date: $(date)" >> experiments.log
echo "Command: python train_with_cross_modal_attention.py ..." >> experiments.log

# 运行训练
python train_with_cross_modal_attention.py \
    --cross_modal_num_heads 4 \
    --output_dir ./output/exp_heads_4/ \
    2>&1 | tee -a experiments.log
```

---

## 📚 更多资源

- **详细使用文档**: `cross_modal_attention_usage.md`
- **架构说明**: `cross_modal_attention_architecture.md`
- **改进建议**: `improvement_suggestions.md`
- **简单示例**: `simple_training_example.py`
- **Bash脚本**: `train_examples.sh`

---

## 🤝 技术支持

如有问题：

1. 查看 `cross_modal_attention_usage.md` 中的故障排除部分
2. 检查数据集路径和格式是否正确
3. 运行快速测试验证环境配置：
   ```bash
   python simple_training_example.py
   ```

---

## 📝 引用

如果使用本代码，请引用：

```bibtex
@inproceedings{das2023crysmmnet,
  title={CrysMMNet: Multimodal Representation for Crystal Property Prediction},
  author={Das, Kishalay and Goyal, Pawan and Lee, Seung-Cheol and Bhattacharjee, Satadeep and Ganguly, Niloy},
  booktitle={The 39th Conference on Uncertainty in Artificial Intelligence},
  year={2023}
}
```

---

**最后更新**: 2025-11-13
**版本**: 1.0
**作者**: Claude Code
