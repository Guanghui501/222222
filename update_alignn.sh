#!/bin/bash
# 自动更新 alignn.py 到支持跨模态注意力的版本
# 使用方法: bash update_alignn.sh

set -e

echo "================================================================"
echo "CrysMMNet 跨模态注意力 - 文件更新脚本"
echo "================================================================"
echo ""

# 检查当前目录
if [ ! -f "train_folder.py" ]; then
    echo "❌ 错误: 请在 crysmmnet-main/src 目录下运行此脚本"
    echo "当前目录: $(pwd)"
    echo ""
    echo "请执行："
    echo "  cd /public/home/ghzhang/crysmmnet-main/src"
    echo "  bash update_alignn.sh"
    exit 1
fi

echo "✓ 当前目录正确"
echo ""

# 备份原文件
if [ -f "models/alignn.py" ]; then
    echo "备份原文件..."
    cp models/alignn.py models/alignn.py.backup.$(date +%Y%m%d_%H%M%S)
    echo "✓ 原文件已备份"
else
    echo "⚠️  警告: 原 alignn.py 不存在"
fi
echo ""

# GitHub 仓库信息
GITHUB_RAW_URL="https://raw.githubusercontent.com/Guanghui501/222222"
BRANCH="claude/parse-functionality-011CV5SzJtxst49fCeaEgG5G"
FILE_PATH="crysmmnet-main/src/models/alignn.py"

DOWNLOAD_URL="${GITHUB_RAW_URL}/${BRANCH}/${FILE_PATH}"

echo "准备下载更新文件..."
echo "URL: ${DOWNLOAD_URL}"
echo ""

# 尝试下载
if command -v curl &> /dev/null; then
    echo "使用 curl 下载..."
    if curl -f -o models/alignn.py.new "${DOWNLOAD_URL}"; then
        echo "✓ 下载成功"
        mv models/alignn.py.new models/alignn.py
        echo "✓ 文件已更新"
    else
        echo "❌ 下载失败（curl）"
        echo ""
        echo "可能原因："
        echo "  1. 服务器无法访问 GitHub"
        echo "  2. 网络连接问题"
        echo ""
        echo "请使用手动方式更新（见下方说明）"
        exit 1
    fi
elif command -v wget &> /dev/null; then
    echo "使用 wget 下载..."
    if wget -O models/alignn.py.new "${DOWNLOAD_URL}"; then
        echo "✓ 下载成功"
        mv models/alignn.py.new models/alignn.py
        echo "✓ 文件已更新"
    else
        echo "❌ 下载失败（wget）"
        echo ""
        echo "可能原因："
        echo "  1. 服务器无法访问 GitHub"
        echo "  2. 网络连接问题"
        echo ""
        echo "请使用手动方式更新（见下方说明）"
        exit 1
    fi
else
    echo "❌ 错误: 系统中没有 curl 或 wget"
    echo ""
    echo "请使用手动方式更新（见下方说明）"
    exit 1
fi

echo ""
echo "================================================================"
echo "验证更新..."
echo "================================================================"

# 验证文件
if grep -q "class CrossModalAttention" models/alignn.py; then
    echo "✓ CrossModalAttention 类存在"
else
    echo "❌ 错误: CrossModalAttention 类未找到"
    echo "文件可能损坏，请检查"
    exit 1
fi

if grep -q "use_cross_modal_attention" models/alignn.py; then
    echo "✓ 跨模态注意力配置参数存在"
else
    echo "❌ 错误: 配置参数未找到"
    echo "文件可能损坏，请检查"
    exit 1
fi

echo ""
echo "================================================================"
echo "更新完成！"
echo "================================================================"
echo ""
echo "更新的文件:"
echo "  - models/alignn.py (已更新)"
echo ""
echo "备份文件:"
echo "  - models/alignn.py.backup.*"
echo ""
echo "现在可以运行训练了："
echo ""
echo "  python train_with_cross_modal_attention.py \\"
echo "      --dataset jarvis \\"
echo "      --property mbj_bandgap \\"
echo "      --epochs 1000 \\"
echo "      --batch_size 64"
echo ""
echo "或使用原训练脚本："
echo ""
echo "  python train_folder.py \\"
echo "      --root_dir '../dataset/' \\"
echo "      --dataset 'Jarvis' \\"
echo "      --property 'mbj_bandgap' \\"
echo "      --epochs 1000 \\"
echo "      --batch_size 64"
echo ""

# 显示备份文件
echo "备份文件列表："
ls -lh models/alignn.py.backup.* 2>/dev/null || echo "  (无备份文件)"
echo ""

echo "✅ 所有操作完成！"
