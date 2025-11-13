#!/bin/bash

echo "=========================================="
echo "CrysMMNet 修复代码下载脚本"
echo "=========================================="
echo ""

# 配置
GITHUB_USER="Guanghui501"
REPO="222222"
BRANCH="claude/parse-functionality-011CV5SzJtxst49fCeaEgG5G"
BASE_URL="https://raw.githubusercontent.com/${GITHUB_USER}/${REPO}/${BRANCH}"

# 检查 curl 是否可用
if ! command -v curl &> /dev/null; then
    echo "❌ 错误: 未找到 curl 命令"
    echo "   请安装 curl: sudo apt-get install curl (Ubuntu/Debian)"
    echo "                或 brew install curl (macOS)"
    exit 1
fi

# 创建目录
echo "📁 创建目录结构..."
mkdir -p crysmmnet-main/src/models
mkdir -p backup_original

# 备份原文件（如果存在）
echo ""
echo "📦 备份原始文件..."
if [ -f "crysmmnet-main/src/models/alignn.py" ]; then
    cp crysmmnet-main/src/models/alignn.py backup_original/alignn.py.bak
    echo "  ✓ 已备份 alignn.py"
fi
if [ -f "train_with_cross_modal_attention.py" ]; then
    cp train_with_cross_modal_attention.py backup_original/train_with_cross_modal_attention.py.bak
    echo "  ✓ 已备份 train_with_cross_modal_attention.py"
fi

# 下载文件
echo ""
echo "📥 下载修复后的文件..."

FILES=(
    "crysmmnet-main/src/models/alignn.py"
    "train_with_cross_modal_attention.py"
    "contrastive_trainer.py"
    "analyze_results.py"
    "MIDDLE_FUSION_FIX.md"
    "PERFORMANCE_IMPROVEMENT_GUIDE.md"
    "CONTRASTIVE_LEARNING_GUIDE.md"
)

SUCCESS=0
FAILED=0

for file in "${FILES[@]}"; do
    echo -n "  下载 ${file}..."
    if curl -f -s -o "$file" "${BASE_URL}/${file}"; then
        echo " ✓"
        ((SUCCESS++))
    else
        echo " ✗ 失败"
        ((FAILED++))
    fi
done

echo ""
echo "=========================================="
echo "下载完成: ${SUCCESS} 成功, ${FAILED} 失败"
echo "=========================================="

if [ $FAILED -eq 0 ]; then
    echo ""
    echo "✓ 所有文件下载成功！"
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
else
    echo ""
    echo "⚠️  部分文件下载失败"
    echo "请检查："
    echo "  1. 网络连接是否正常"
    echo "  2. GitHub 仓库地址是否正确: https://github.com/${GITHUB_USER}/${REPO}"
    echo "  3. 分支名称是否正确: ${BRANCH}"
    echo ""
    echo "或者使用 git 命令下载："
    echo "  git fetch origin ${BRANCH}"
    echo "  git checkout ${BRANCH}"
fi
