#!/bin/bash

# ================================================================
# 融合策略对比实验脚本
# ================================================================
# 用途：对比简单拼接、晚期融合、中期融合、中期+晚期融合的性能
# 使用方法: bash run_fusion_comparison.sh
# ================================================================

# 配置参数
DATASET="jarvis"
PROPERTY="mbj_bandgap"
EPOCHS=100
BATCH_SIZE=64
LEARNING_RATE=0.001

# 可选：快速测试（小数据集，少epoch）
# 取消下面这些注释来进行快速测试
# EPOCHS=10
# N_TRAIN="--n_train 1000 --n_val 100 --n_test 100"

echo "================================================================"
echo "融合策略对比实验"
echo "================================================================"
echo "数据集: $DATASET"
echo "性质: $PROPERTY"
echo "训练轮数: $EPOCHS"
echo "批次大小: $BATCH_SIZE"
echo "================================================================"
echo ""

# ================================================================
# 实验1: 基线（简单拼接，不使用任何注意力机制）
# ================================================================
echo "================================================================"
echo "实验1/4: 基线（简单拼接）"
echo "================================================================"
python train_with_cross_modal_attention.py \
    --dataset $DATASET \
    --property $PROPERTY \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --use_cross_modal False \
    --use_middle_fusion False \
    --output_dir ./output_comparison/baseline/ \
    $N_TRAIN

echo ""
echo "✓ 实验1完成"
echo ""

# ================================================================
# 实验2: 晚期融合（跨模态注意力）
# ================================================================
echo "================================================================"
echo "实验2/4: 晚期融合（跨模态注意力）"
echo "================================================================"
python train_with_cross_modal_attention.py \
    --dataset $DATASET \
    --property $PROPERTY \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --use_middle_fusion False \
    --output_dir ./output_comparison/late_fusion/ \
    $N_TRAIN

echo ""
echo "✓ 实验2完成"
echo ""

# ================================================================
# 实验3: 中期融合
# ================================================================
echo "================================================================"
echo "实验3/4: 中期融合（在ALIGNN第2层后注入）"
echo "================================================================"
python train_with_cross_modal_attention.py \
    --dataset $DATASET \
    --property $PROPERTY \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --use_cross_modal False \
    --use_middle_fusion True \
    --middle_fusion_layers "2" \
    --middle_fusion_num_heads 4 \
    --middle_fusion_hidden_dim 256 \
    --output_dir ./output_comparison/middle_fusion/ \
    $N_TRAIN

echo ""
echo "✓ 实验3完成"
echo ""

# ================================================================
# 实验4: 中期融合 + 晚期融合（组合方案）
# ================================================================
echo "================================================================"
echo "实验4/4: 中期融合 + 晚期融合（组合方案）"
echo "================================================================"
python train_with_cross_modal_attention.py \
    --dataset $DATASET \
    --property $PROPERTY \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --use_middle_fusion True \
    --middle_fusion_layers "2" \
    --middle_fusion_num_heads 2 \
    --middle_fusion_hidden_dim 128 \
    --output_dir ./output_comparison/middle_late_fusion/ \
    $N_TRAIN

echo ""
echo "✓ 实验4完成"
echo ""

# ================================================================
# 收集结果
# ================================================================
echo "================================================================"
echo "所有实验完成！"
echo "================================================================"
echo ""
echo "结果保存在以下目录："
echo "  1. 基线:             ./output_comparison/baseline/$PROPERTY/"
echo "  2. 晚期融合:         ./output_comparison/late_fusion/$PROPERTY/"
echo "  3. 中期融合:         ./output_comparison/middle_fusion/$PROPERTY/"
echo "  4. 中期+晚期融合:    ./output_comparison/middle_late_fusion/$PROPERTY/"
echo ""
echo "查看结果："
echo "  训练指标: tail -20 ./output_comparison/*/mbj_bandgap/train_metrics.csv"
echo "  验证指标: tail -20 ./output_comparison/*/mbj_bandgap/val_metrics.csv"
echo "  测试指标: cat ./output_comparison/*/mbj_bandgap/test_metrics.csv"
echo ""
echo "================================================================"
echo "结果对比脚本"
echo "================================================================"

# 创建结果对比Python脚本
cat > compare_results.py << 'EOF'
import pandas as pd
import matplotlib.pyplot as plt
import os

# 配置
experiments = {
    'Baseline (Simple Concat)': './output_comparison/baseline/mbj_bandgap/',
    'Late Fusion (Cross-Modal)': './output_comparison/late_fusion/mbj_bandgap/',
    'Middle Fusion': './output_comparison/middle_fusion/mbj_bandgap/',
    'Middle + Late Fusion': './output_comparison/middle_late_fusion/mbj_bandgap/',
}

# 读取验证集指标
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Fusion Strategy Comparison', fontsize=16)

for name, path in experiments.items():
    val_file = os.path.join(path, 'val_metrics.csv')
    if os.path.exists(val_file):
        df = pd.read_csv(val_file)

        # 绘制MAE
        axes[0, 0].plot(df['epoch'], df['mae'], label=name, marker='o', markersize=3)

        # 绘制RMSE
        axes[0, 1].plot(df['epoch'], df.get('rmse', df.get('loss')), label=name, marker='o', markersize=3)

        # 绘制Loss
        axes[1, 0].plot(df['epoch'], df['loss'], label=name, marker='o', markersize=3)

        # 打印最终性能
        print(f"\n{name}:")
        print(f"  Final MAE: {df['mae'].iloc[-1]:.4f}")
        print(f"  Best MAE: {df['mae'].min():.4f}")
        print(f"  Final Loss: {df['loss'].iloc[-1]:.4f}")

# 设置图表
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('MAE')
axes[0, 0].set_title('Validation MAE')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('RMSE')
axes[0, 1].set_title('Validation RMSE')
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].set_title('Validation Loss')
axes[1, 0].legend()
axes[1, 0].grid(True)

# 读取测试集结果
axes[1, 1].axis('off')
summary_text = "Test Set Results\n" + "="*40 + "\n"

for name, path in experiments.items():
    test_file = os.path.join(path, 'test_metrics.csv')
    if os.path.exists(test_file):
        df = pd.read_csv(test_file)
        if len(df) > 0:
            summary_text += f"\n{name}:\n"
            summary_text += f"  MAE: {df['mae'].iloc[0]:.4f}\n"
            if 'rmse' in df.columns:
                summary_text += f"  RMSE: {df['rmse'].iloc[0]:.4f}\n"

axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')

plt.tight_layout()
plt.savefig('fusion_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n图表已保存到: fusion_comparison.png")

# 创建CSV摘要
summary_data = []
for name, path in experiments.items():
    val_file = os.path.join(path, 'val_metrics.csv')
    test_file = os.path.join(path, 'test_metrics.csv')

    row = {'Method': name}

    if os.path.exists(val_file):
        df = pd.read_csv(val_file)
        row['Val_Final_MAE'] = df['mae'].iloc[-1]
        row['Val_Best_MAE'] = df['mae'].min()
        row['Val_Final_Loss'] = df['loss'].iloc[-1]

    if os.path.exists(test_file):
        df = pd.read_csv(test_file)
        if len(df) > 0:
            row['Test_MAE'] = df['mae'].iloc[0]
            if 'rmse' in df.columns:
                row['Test_RMSE'] = df['rmse'].iloc[0]

    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('fusion_comparison_summary.csv', index=False)
print(f"摘要已保存到: fusion_comparison_summary.csv")
print("\n" + "="*60)
print(summary_df.to_string(index=False))
EOF

echo ""
echo "运行结果分析脚本："
echo "  python compare_results.py"
echo ""
echo "这将生成："
echo "  - fusion_comparison.png: 性能对比图"
echo "  - fusion_comparison_summary.csv: 结果摘要表"
echo ""
