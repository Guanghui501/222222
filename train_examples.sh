#!/bin/bash
# ============================================================================
# CrysMMNet 训练示例脚本集合
# 包含多种训练场景的示例
# ============================================================================

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 设置Python环境（如果使用虚拟环境）
# source venv/bin/activate

# ============================================================================
# 1. 基础训练示例
# ============================================================================

echo "示例 1: JARVIS 形成能训练（使用跨模态注意力）"
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --epochs 1000 \
    --batch_size 64 \
    --output_dir ./output/jarvis_fe_cross_modal/

# ============================================================================
# 2. 对比实验：使用 vs 不使用跨模态注意力
# ============================================================================

echo ""
echo "示例 2a: 不使用跨模态注意力（baseline）"
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy \
    --use_cross_modal False \
    --epochs 1000 \
    --batch_size 64 \
    --output_dir ./output/jarvis_fe_baseline/

echo ""
echo "示例 2b: 使用跨模态注意力"
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --epochs 1000 \
    --batch_size 64 \
    --output_dir ./output/jarvis_fe_cross_modal/

# ============================================================================
# 3. 不同注意力头数的消融实验
# ============================================================================

echo ""
echo "示例 3a: 2个注意力头"
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy \
    --use_cross_modal True \
    --cross_modal_num_heads 2 \
    --epochs 1000 \
    --batch_size 64 \
    --output_dir ./output/jarvis_fe_heads_2/

echo ""
echo "示例 3b: 4个注意力头（推荐）"
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --epochs 1000 \
    --batch_size 64 \
    --output_dir ./output/jarvis_fe_heads_4/

echo ""
echo "示例 3c: 8个注意力头"
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy \
    --use_cross_modal True \
    --cross_modal_num_heads 8 \
    --epochs 1000 \
    --batch_size 64 \
    --output_dir ./output/jarvis_fe_heads_8/

# ============================================================================
# 4. JARVIS 数据集 - 所有性质训练
# ============================================================================

echo ""
echo "示例 4: JARVIS 所有性质"

# 形成能
python train_with_cross_modal_attention.py \
    --dataset jarvis --property formation_energy \
    --use_cross_modal True --cross_modal_num_heads 4 \
    --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 \
    --epochs 1000 --batch_size 64

# 总能量
python train_with_cross_modal_attention.py \
    --dataset jarvis --property total_energy \
    --use_cross_modal True --cross_modal_num_heads 4 \
    --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 \
    --epochs 1000 --batch_size 64

# OPT 带隙
python train_with_cross_modal_attention.py \
    --dataset jarvis --property opt_bandgap \
    --use_cross_modal True --cross_modal_num_heads 4 \
    --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 \
    --epochs 1000 --batch_size 64

# MBJ 带隙
python train_with_cross_modal_attention.py \
    --dataset jarvis --property mbj_bandgap \
    --use_cross_modal True --cross_modal_num_heads 4 \
    --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 \
    --epochs 1000 --batch_size 64

# 体积模量
python train_with_cross_modal_attention.py \
    --dataset jarvis --property bulk_modulus_kv \
    --use_cross_modal True --cross_modal_num_heads 4 \
    --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 \
    --epochs 1000 --batch_size 64

# 剪切模量
python train_with_cross_modal_attention.py \
    --dataset jarvis --property shear_modulus_gv \
    --use_cross_modal True --cross_modal_num_heads 4 \
    --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 \
    --epochs 1000 --batch_size 64

# ============================================================================
# 5. Material Project 数据集训练
# ============================================================================

echo ""
echo "示例 5: Material Project 数据集"

# 形成能
python train_with_cross_modal_attention.py \
    --dataset mp --property formation_energy \
    --use_cross_modal True --cross_modal_num_heads 4 \
    --n_train 60000 --n_val 5000 --n_test 4132 \
    --epochs 1000 --batch_size 64

# 带隙
python train_with_cross_modal_attention.py \
    --dataset mp --property band_gap \
    --use_cross_modal True --cross_modal_num_heads 4 \
    --n_train 60000 --n_val 5000 --n_test 4132 \
    --epochs 1000 --batch_size 64

# 体积模量
python train_with_cross_modal_attention.py \
    --dataset mp --property bulk \
    --use_cross_modal True --cross_modal_num_heads 4 \
    --n_train 4664 --n_val 393 --n_test 393 \
    --epochs 1000 --batch_size 64

# 剪切模量
python train_with_cross_modal_attention.py \
    --dataset mp --property shear \
    --use_cross_modal True --cross_modal_num_heads 4 \
    --n_train 4664 --n_val 393 --n_test 393 \
    --epochs 1000 --batch_size 64

# ============================================================================
# 6. 不同隐藏维度的实验
# ============================================================================

echo ""
echo "示例 6: 不同隐藏维度"

# 小模型 (128维)
python train_with_cross_modal_attention.py \
    --dataset jarvis --property formation_energy \
    --use_cross_modal True \
    --cross_modal_hidden_dim 128 \
    --cross_modal_num_heads 4 \
    --epochs 1000 --batch_size 64 \
    --output_dir ./output/jarvis_fe_dim_128/

# 标准模型 (256维)
python train_with_cross_modal_attention.py \
    --dataset jarvis --property formation_energy \
    --use_cross_modal True \
    --cross_modal_hidden_dim 256 \
    --cross_modal_num_heads 4 \
    --epochs 1000 --batch_size 64 \
    --output_dir ./output/jarvis_fe_dim_256/

# 大模型 (512维)
python train_with_cross_modal_attention.py \
    --dataset jarvis --property formation_energy \
    --use_cross_modal True \
    --cross_modal_hidden_dim 512 \
    --cross_modal_num_heads 8 \
    --epochs 1000 --batch_size 64 \
    --output_dir ./output/jarvis_fe_dim_512/

# ============================================================================
# 7. 不同Dropout率的实验
# ============================================================================

echo ""
echo "示例 7: 不同Dropout率"

# 低dropout (0.05)
python train_with_cross_modal_attention.py \
    --dataset jarvis --property formation_energy \
    --use_cross_modal True \
    --cross_modal_dropout 0.05 \
    --epochs 1000 --batch_size 64 \
    --output_dir ./output/jarvis_fe_dropout_005/

# 中等dropout (0.1，推荐)
python train_with_cross_modal_attention.py \
    --dataset jarvis --property formation_energy \
    --use_cross_modal True \
    --cross_modal_dropout 0.1 \
    --epochs 1000 --batch_size 64 \
    --output_dir ./output/jarvis_fe_dropout_01/

# 高dropout (0.2)
python train_with_cross_modal_attention.py \
    --dataset jarvis --property formation_energy \
    --use_cross_modal True \
    --cross_modal_dropout 0.2 \
    --epochs 1000 --batch_size 64 \
    --output_dir ./output/jarvis_fe_dropout_02/

# ============================================================================
# 8. 快速测试（小数据集，少epoch）
# ============================================================================

echo ""
echo "示例 8: 快速测试"
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --n_train 1000 \
    --n_val 100 \
    --n_test 100 \
    --epochs 10 \
    --batch_size 32 \
    --output_dir ./output/quick_test/

# ============================================================================
# 9. 从检查点恢复训练
# ============================================================================

echo ""
echo "示例 9: 从检查点恢复训练"
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --epochs 1000 \
    --batch_size 64 \
    --resume 1 \
    --output_dir ./output/jarvis_fe_resume/

# ============================================================================
# 10. GPU内存优化配置（显存有限时使用）
# ============================================================================

echo ""
echo "示例 10: GPU内存优化配置"
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy \
    --use_cross_modal True \
    --cross_modal_hidden_dim 128 \
    --cross_modal_num_heads 2 \
    --hidden_features 128 \
    --epochs 1000 \
    --batch_size 32 \
    --output_dir ./output/jarvis_fe_lowmem/

echo ""
echo "所有训练示例执行完毕！"
