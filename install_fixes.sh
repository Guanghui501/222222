#!/bin/bash

echo "=========================================="
echo "CrysMMNet 修复代码安装脚本"
echo "=========================================="
echo ""

# 检查是否在正确的目录
if [ ! -d "crysmmnet-main" ]; then
    echo "❌ 错误: 请在 222222 项目根目录下运行此脚本"
    echo "   当前目录: $(pwd)"
    exit 1
fi

echo "✓ 检测到项目目录"
echo ""

# 备份原文件
echo "📦 备份原始文件..."
backup_dir="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

if [ -f "crysmmnet-main/src/models/alignn.py" ]; then
    cp crysmmnet-main/src/models/alignn.py "$backup_dir/"
    echo "  - 已备份 alignn.py"
fi

if [ -f "train_with_cross_modal_attention.py" ]; then
    cp train_with_cross_modal_attention.py "$backup_dir/"
    echo "  - 已备份 train_with_cross_modal_attention.py"
fi

echo "  备份保存在: $backup_dir/"
echo ""

# 应用补丁
if [ -f "all_fixes.patch" ]; then
    echo "🔧 应用修复补丁..."
    if git apply --check all_fixes.patch 2>/dev/null; then
        git apply all_fixes.patch
        echo "✓ 补丁应用成功"
    else
        echo "⚠️  补丁应用失败，尝试直接复制文件..."
        # 补丁失败时的说明
        echo "   请手动使用压缩包中的文件替换"
    fi
else
    echo "ℹ️  未找到补丁文件，文件已直接包含在压缩包中"
fi

echo ""
echo "=========================================="
echo "✓ 安装完成！"
echo "=========================================="
echo ""
echo "📄 修复内容："
echo "  1. 修复中期融合张量形状错误 (commit 1ef7a66)"
echo "  2. 修复后期融合信息丢失问题 (commit 33bc288)"
echo "  3. 添加性能改进指南和分析工具 (commit 15d9d2e)"
echo "  4. 添加对比学习支持 (commit d91b1e2)"
echo ""
echo "📚 使用文档："
echo "  - MIDDLE_FUSION_FIX.md - 中期融合修复说明"
echo "  - PERFORMANCE_IMPROVEMENT_GUIDE.md - 性能改进指南"
echo "  - CONTRASTIVE_LEARNING_GUIDE.md - 对比学习使用文档"
echo ""
echo "🚀 快速测试命令："
echo "  python train_with_cross_modal_attention.py \\"
echo "    --dataset jarvis \\"
echo "    --property mbj_bandgap \\"
echo "    --use_cross_modal True \\"
echo "    --use_contrastive True \\"
echo "    --epochs 100 \\"
echo "    --batch_size 64"
echo ""
