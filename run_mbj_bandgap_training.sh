#!/bin/bash
# 专门为 MBJ Bandgap 训练准备的快速启动脚本
# 使用方法: bash run_mbj_bandgap_training.sh

set -e  # 遇到错误立即退出

echo "=================================================="
echo "JARVIS MBJ Bandgap 训练 - 跨模态注意力"
echo "=================================================="

# 检查是否在正确的目录
if [ ! -f "train_with_cross_modal_attention.py" ]; then
    echo "错误: 找不到训练脚本"
    echo "请先 cd 到 crysmmnet-main/src 目录"
    exit 1
fi

# 检查数据集是否存在
if [ ! -d "../dataset/jarvis/mbj_bandgap" ]; then
    echo "错误: 找不到数据集目录 ../dataset/jarvis/mbj_bandgap"
    echo "请确保数据集已下载并解压到正确位置"
    exit 1
fi

# 检查 vocab_mappings.txt
if [ ! -f "vocab_mappings.txt" ]; then
    echo "错误: 找不到 vocab_mappings.txt"
    echo "请确保该文件存在于当前目录"
    exit 1
fi

echo ""
echo "✓ 环境检查通过"
echo "✓ 训练脚本: train_with_cross_modal_attention.py"
echo "✓ 数据集: ../dataset/jarvis/mbj_bandgap"
echo "✓ 词汇映射: vocab_mappings.txt"
echo ""

# 设置训练参数
DATASET="jarvis"
PROPERTY="mbj_bandgap"
USE_CROSS_MODAL="True"
NUM_HEADS=4
EPOCHS=1000
BATCH_SIZE=64
OUTPUT_DIR="./output/mbj_bandgap_cross_modal"

echo "训练配置:"
echo "  数据集: $DATASET"
echo "  性质: $PROPERTY"
echo "  跨模态注意力: $USE_CROSS_MODAL"
echo "  注意力头数: $NUM_HEADS"
echo "  训练轮数: $EPOCHS"
echo "  批次大小: $BATCH_SIZE"
echo "  输出目录: $OUTPUT_DIR"
echo ""

read -p "确认开始训练? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "训练已取消"
    exit 0
fi

echo ""
echo "=================================================="
echo "开始训练..."
echo "=================================================="
echo ""

# 运行训练
python train_with_cross_modal_attention.py \
    --dataset "$DATASET" \
    --property "$PROPERTY" \
    --use_cross_modal "$USE_CROSS_MODAL" \
    --cross_modal_num_heads "$NUM_HEADS" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$OUTPUT_DIR" \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1

echo ""
echo "=================================================="
echo "训练完成！"
echo "=================================================="
echo ""
echo "结果保存在: $OUTPUT_DIR"
echo ""
echo "查看结果:"
echo "  - 配置: $OUTPUT_DIR/config.json"
echo "  - 最佳模型: $OUTPUT_DIR/checkpoint_best.pt"
echo "  - 测试预测: $OUTPUT_DIR/test_predictions.csv"
echo "  - 训练历史: $OUTPUT_DIR/history.json"
echo ""
