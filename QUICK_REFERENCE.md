# 快速参考卡 - CrysMMNet 跨模态注意力

## 🚀 30秒快速开始

```bash
# 1. 查看帮助
python train_with_cross_modal_attention.py --help

# 2. 开始训练
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy \
    --epochs 1000 \
    --batch_size 64
```

## 📋 常用命令

### JARVIS 数据集

```bash
# 形成能
python train_with_cross_modal_attention.py \
    --dataset jarvis --property formation_energy \
    --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1

# 带隙
python train_with_cross_modal_attention.py \
    --dataset jarvis --property opt_bandgap

# 体积模量
python train_with_cross_modal_attention.py \
    --dataset jarvis --property bulk_modulus_kv
```

### Material Project 数据集

```bash
# 形成能
python train_with_cross_modal_attention.py \
    --dataset mp --property formation_energy \
    --n_train 60000 --n_val 5000 --n_test 4132

# 带隙
python train_with_cross_modal_attention.py \
    --dataset mp --property band_gap \
    --n_train 60000 --n_val 5000 --n_test 4132
```

## 🎛️ 核心参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--use_cross_modal` | `True` | 启用跨模态注意力 |
| `--cross_modal_num_heads` | `4` | 注意力头数 (1/2/4/8) |
| `--cross_modal_hidden_dim` | `256` | 隐藏维度 |
| `--cross_modal_dropout` | `0.1` | Dropout率 |
| `--batch_size` | `64` | 批次大小 |
| `--epochs` | `1000` | 训练轮数 |
| `--learning_rate` | `0.001` | 学习率 |

## 🧪 对比实验模板

```bash
# Baseline
python train_with_cross_modal_attention.py \
    --use_cross_modal False \
    --output_dir ./output/baseline/

# 跨模态注意力
python train_with_cross_modal_attention.py \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --output_dir ./output/cross_modal/
```

## 🔧 不同场景配置

### 标准配置（推荐）
```bash
--use_cross_modal True \
--cross_modal_hidden_dim 256 \
--cross_modal_num_heads 4 \
--batch_size 64
```

### 显存有限
```bash
--hidden_features 128 \
--cross_modal_hidden_dim 128 \
--cross_modal_num_heads 2 \
--batch_size 32
```

### 高性能
```bash
--hidden_features 512 \
--cross_modal_hidden_dim 512 \
--cross_modal_num_heads 8 \
--batch_size 64
```

### 快速测试
```bash
--n_train 1000 --n_val 100 --n_test 100 \
--epochs 10 --batch_size 32
```

## 📊 支持的性质

### JARVIS-DFT
- `formation_energy` / `fe`
- `total_energy`
- `opt_bandgap`
- `mbj_bandgap`
- `bulk_modulus_kv`
- `shear_modulus_gv`

### Material Project
- `formation_energy`
- `band_gap`
- `bulk`
- `shear`

## 🐛 常见问题速查

| 问题 | 解决方案 |
|------|---------|
| CUDA OOM | `--batch_size 32 --cross_modal_num_heads 2` |
| 维度错误 | 确保 `hidden_dim % num_heads == 0` |
| 训练慢 | `--epochs 100` 或 `--n_train 5000` |
| 路径错误 | 检查 `--root_dir` 是否正确 |

## 📁 输出文件

```
output/formation_energy/
├── config.json              # 配置
├── checkpoint_best.pt       # 最佳模型
├── test_predictions.csv     # 测试结果
└── history.json            # 训练历史
```

## 💡 提示

- ✅ 默认启用跨模态注意力
- ✅ 使用 `--help` 查看所有参数
- ✅ 使用 `--resume 1` 从checkpoint恢复
- ✅ 查看 `TRAINING_GUIDE.md` 获取详细说明

## 📚 文档

- **训练指南**: `TRAINING_GUIDE.md`
- **使用手册**: `cross_modal_attention_usage.md`
- **架构说明**: `cross_modal_attention_architecture.md`
- **改进建议**: `improvement_suggestions.md`

---

**快速参考 | 2025-11-13**
