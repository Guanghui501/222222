#!/usr/bin/env python
"""
CrysMMNet 预测可解释性分析脚本

功能：
1. 加载训练好的模型
2. 分析测试集样本的预测
3. 可视化原子重要性、注意力权重、特征空间等
4. 生成可解释性报告

使用方法：
    python explain_predictions.py \
        --model_path ./output/best_model.pt \
        --dataset jarvis \
        --property mbj_bandgap \
        --num_samples 10 \
        --output_dir ./interpretability_results/
"""

import argparse
import torch
import os
import sys
import numpy as np
from pathlib import Path

# 添加路径
sys.path.insert(0, 'crysmmnet-main/src')

from interpretability import InterpretabilityAnalyzer, create_interpretability_report
from models.alignn import ALIGNN, ALIGNNConfig
from jarvis.core.atoms import Atoms
from graphs import Graph
import dgl


def load_model(model_path, config_path=None, device='cuda'):
    """
    加载训练好的模型

    Args:
        model_path: 模型checkpoint路径
        config_path: 配置文件路径（可选）
        device: 设备

    Returns:
        model: 加载的模型
        config: 模型配置
    """
    print(f"\n加载模型: {model_path}")

    # 加载checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # 尝试从checkpoint中获取配置
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = ALIGNNConfig(**config_dict)
    elif config_path and os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = ALIGNNConfig(**config_dict)
    else:
        # 使用默认配置
        print("警告: 未找到模型配置，使用默认配置")
        config = ALIGNNConfig()

    # 创建模型
    model = ALIGNN(config)

    # 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print(f"✓ 模型加载成功")
    print(f"  - 参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - 跨模态注意力: {model.use_cross_modal_attention}")
    print(f"  - 对比学习: {model.use_contrastive_loss}")

    return model, config


def analyze_single_sample(analyzer, cif_file, text_description, true_value=None,
                         save_dir=None, sample_id='sample'):
    """
    分析单个样本的预测

    Args:
        analyzer: InterpretabilityAnalyzer实例
        cif_file: CIF文件路径
        text_description: 文本描述
        true_value: 真实值（可选）
        save_dir: 保存目录
        sample_id: 样本ID

    Returns:
        explanation: 解释字典
    """
    print(f"\n{'='*80}")
    print(f"分析样本: {sample_id}")
    print(f"{'='*80}")

    # 加载结构
    atoms = Atoms.from_cif(cif_file)
    print(f"晶体结构:")
    print(f"  - 化学式: {atoms.composition.reduced_formula}")
    print(f"  - 原子数: {len(atoms)}")
    print(f"  - 空间群: {atoms.spacegroup()}")

    # 构建图
    graph_data = Graph.atom_dgl_multigraph(
        atoms,
        cutoff=8.0,
        atom_features="cgcnn",
        max_neighbors=12,
        compute_line_graph=True,
        use_canonize=True,
    )
    g, lg = graph_data

    # 添加文本（需要批次化）
    g = dgl.batch([g])
    lg = dgl.batch([lg])

    # 创建输入
    graph_input = (g, lg, [text_description])

    # 获取预测和注意力权重
    print(f"\n获取模型预测和注意力权重...")
    with torch.no_grad():
        output = analyzer.model(graph_input, return_features=True, return_attention=True)

    prediction = output['predictions'].cpu().item()
    print(f"\n预测结果:")
    print(f"  - 预测值: {prediction:.4f}")
    if true_value is not None:
        error = abs(prediction - true_value)
        print(f"  - 真实值: {true_value:.4f}")
        print(f"  - 误差: {error:.4f} ({error/true_value*100:.2f}%)")

    # 计算原子重要性
    print(f"\n计算原子重要性...")
    atom_importance = analyzer.compute_atom_importance(graph_input, method='gradient')

    # 可视化原子重要性
    if save_dir:
        sample_dir = Path(save_dir) / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        # 原子重要性可视化
        atom_df = analyzer.visualize_atom_importance(
            atoms, atom_importance,
            save_path=sample_dir / 'atom_importance.png',
            top_k=10
        )

        # 如果有注意力权重，可视化它们
        if 'attention_weights' in output and output['attention_weights']:
            print(f"\n可视化跨模态注意力...")
            attn_weights = output['attention_weights']

            # Graph-to-Text attention
            if 'graph_to_text' in attn_weights:
                g2t_attn = attn_weights['graph_to_text'].cpu().numpy()
                # 平均所有注意力头
                g2t_attn_avg = g2t_attn.mean(axis=1)  # [batch, seq, seq]

                analyzer.visualize_cross_modal_attention(
                    g,
                    g2t_attn_avg,
                    atom_symbols=[atom for atom in atoms.elements],
                    save_path=sample_dir / 'attention_graph_to_text.png'
                )

            # Text-to-Graph attention
            if 'text_to_graph' in attn_weights:
                t2g_attn = attn_weights['text_to_graph'].cpu().numpy()
                t2g_attn_avg = t2g_attn.mean(axis=1)

                analyzer.visualize_cross_modal_attention(
                    g,
                    t2g_attn_avg,
                    atom_symbols=[atom for atom in atoms.elements],
                    save_path=sample_dir / 'attention_text_to_graph.png'
                )

    # 创建解释字典
    explanation = {
        'sample_id': sample_id,
        'formula': atoms.composition.reduced_formula,
        'num_atoms': len(atoms),
        'prediction': prediction,
        'true_value': true_value,
        'error': abs(prediction - true_value) if true_value else None,
        'atom_importance': atom_importance.tolist(),
        'text_description': text_description,
    }

    # 保存解释
    if save_dir:
        import json
        with open(sample_dir / 'explanation.json', 'w') as f:
            json.dump(explanation, f, indent=2)
        print(f"\n✓ 解释已保存到: {sample_dir}")

    return explanation


def main():
    parser = argparse.ArgumentParser(
        description='CrysMMNet 预测可解释性分析',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--model_path', type=str, required=True,
                       help='模型checkpoint路径')
    parser.add_argument('--config_path', type=str, default=None,
                       help='模型配置文件路径（JSON）')

    parser.add_argument('--dataset', type=str, default='jarvis',
                       choices=['jarvis', 'mp'],
                       help='数据集名称')
    parser.add_argument('--property', type=str, default='mbj_bandgap',
                       help='预测的性质')

    parser.add_argument('--cif_file', type=str, default=None,
                       help='单个CIF文件路径（分析单个样本）')
    parser.add_argument('--text', type=str, default=None,
                       help='文本描述（分析单个样本时需要）')
    parser.add_argument('--true_value', type=float, default=None,
                       help='真实值（可选）')

    parser.add_argument('--num_samples', type=int, default=5,
                       help='分析的测试样本数量')
    parser.add_argument('--output_dir', type=str, default='./interpretability_results',
                       help='输出目录')

    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='计算设备')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"CrysMMNet 预测可解释性分析")
    print(f"{'='*80}\n")

    # 加载模型
    model, config = load_model(args.model_path, args.config_path, args.device)

    # 创建分析器
    analyzer = InterpretabilityAnalyzer(model, device=args.device)

    # 如果提供了单个CIF文件，分析它
    if args.cif_file:
        if not args.text:
            print("错误: 分析单个样本时需要提供文本描述 (--text)")
            return

        explanation = analyze_single_sample(
            analyzer,
            args.cif_file,
            args.text,
            args.true_value,
            save_dir=output_dir,
            sample_id='custom_sample'
        )

    else:
        # 分析测试集样本
        print(f"\n加载测试数据...")
        print(f"(此功能需要完整的数据加载器，这里仅为示例)")

        # TODO: 加载测试数据加载器
        # test_loader = load_test_data(args.dataset, args.property)
        # create_interpretability_report(analyzer, test_loader, output_dir, args.num_samples)

        print(f"\n提示: 使用 --cif_file 和 --text 参数分析单个样本")

    print(f"\n{'='*80}")
    print(f"分析完成！结果保存在: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
