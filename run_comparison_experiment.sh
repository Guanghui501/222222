#!/bin/bash
# MBJ Bandgap 对比实验：跨模态注意力 vs Baseline
# 使用方法: bash run_comparison_experiment.sh

set -e

echo "========================================================================"
echo "对比实验: 跨模态注意力 vs Baseline"
echo "数据集: JARVIS MBJ Bandgap"
echo "========================================================================"
echo ""

# 环境检查
if [ ! -f "train_with_cross_modal_attention.py" ]; then
    echo "❌ 错误: 找不到训练脚本"
    echo "请先 cd 到 crysmmnet-main/src 目录"
    exit 1
fi

if [ ! -d "../dataset/jarvis/mbj_bandgap" ]; then
    echo "❌ 错误: 找不到数据集"
    exit 1
fi

echo "✅ 环境检查通过"
echo ""

# 训练参数
DATASET="jarvis"
PROPERTY="mbj_bandgap"
EPOCHS=1000
BATCH_SIZE=64

# 实验1: Baseline（不使用跨模态注意力）
echo "========================================================================"
echo "实验 1/4: Baseline（不使用跨模态注意力）"
echo "========================================================================"
python train_with_cross_modal_attention.py \
    --dataset "$DATASET" \
    --property "$PROPERTY" \
    --use_cross_modal False \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --output_dir ./output/mbj_baseline/ \
    --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1

echo ""
echo "✅ 实验 1 完成"
echo ""
sleep 3

# 实验2: 跨模态注意力（2个头）
echo "========================================================================"
echo "实验 2/4: 跨模态注意力（2个头）"
echo "========================================================================"
python train_with_cross_modal_attention.py \
    --dataset "$DATASET" \
    --property "$PROPERTY" \
    --use_cross_modal True \
    --cross_modal_num_heads 2 \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --output_dir ./output/mbj_heads_2/ \
    --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1

echo ""
echo "✅ 实验 2 完成"
echo ""
sleep 3

# 实验3: 跨模态注意力（4个头，推荐）
echo "========================================================================"
echo "实验 3/4: 跨模态注意力（4个头，推荐）"
echo "========================================================================"
python train_with_cross_modal_attention.py \
    --dataset "$DATASET" \
    --property "$PROPERTY" \
    --use_cross_modal True \
    --cross_modal_num_heads 4 \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --output_dir ./output/mbj_heads_4/ \
    --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1

echo ""
echo "✅ 实验 3 完成"
echo ""
sleep 3

# 实验4: 跨模态注意力（8个头）
echo "========================================================================"
echo "实验 4/4: 跨模态注意力（8个头）"
echo "========================================================================"
python train_with_cross_modal_attention.py \
    --dataset "$DATASET" \
    --property "$PROPERTY" \
    --use_cross_modal True \
    --cross_modal_num_heads 8 \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --output_dir ./output/mbj_heads_8/ \
    --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1

echo ""
echo "✅ 实验 4 完成"
echo ""

# 结果总结
echo "========================================================================"
echo "所有实验完成！"
echo "========================================================================"
echo ""
echo "结果位置:"
echo "  1. Baseline:           ./output/mbj_baseline/"
echo "  2. 跨模态注意力(2头): ./output/mbj_heads_2/"
echo "  3. 跨模态注意力(4头): ./output/mbj_heads_4/"
echo "  4. 跨模态注意力(8头): ./output/mbj_heads_8/"
echo ""

# 创建结果对比脚本
cat > compare_results.py << 'EOF'
import json
import pandas as pd

experiments = [
    ('Baseline', './output/mbj_baseline/history.json'),
    ('跨模态(2头)', './output/mbj_heads_2/history.json'),
    ('跨模态(4头)', './output/mbj_heads_4/history.json'),
    ('跨模态(8头)', './output/mbj_heads_8/history.json'),
]

print("=" * 70)
print("MBJ Bandgap 实验结果对比")
print("=" * 70)
print(f"{'实验配置':<20} {'最佳验证MAE':<15} {'最终测试MAE':<15}")
print("-" * 70)

results = []
for name, path in experiments:
    try:
        with open(path, 'r') as f:
            history = json.load(f)
        best_val_mae = min(history['val_mae'])
        final_test_mae = history['test_mae'][-1]
        results.append((name, best_val_mae, final_test_mae))
        print(f"{name:<20} {best_val_mae:<15.4f} {final_test_mae:<15.4f}")
    except FileNotFoundError:
        print(f"{name:<20} {'文件不存在':<15} {'N/A':<15}")

print("=" * 70)

if len(results) > 1:
    baseline_mae = results[0][2]
    print(f"\n相对于Baseline的改进:")
    for name, _, test_mae in results[1:]:
        improvement = (baseline_mae - test_mae) / baseline_mae * 100
        print(f"  {name}: {improvement:+.2f}%")
EOF

echo "运行结果对比:"
python compare_results.py

echo ""
echo "完整对比脚本已保存为: compare_results.py"
echo "可以随时运行: python compare_results.py"
