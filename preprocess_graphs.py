#!/usr/bin/env python
"""
预处理脚本：预先生成和保存图数据

功能：
1. 加载 CIF 文件并构建图
2. 加载文本描述
3. 保存为 pickle 文件供训练时快速加载

使用方法：
    python preprocess_graphs.py --dataset jarvis --property mbj_bandgap

生成的文件：
    preprocessed_data/jarvis_mbj_bandgap_train.pkl
    preprocessed_data/jarvis_mbj_bandgap_val.pkl
    preprocessed_data/jarvis_mbj_bandgap_test.pkl
"""

import argparse
import os
import csv
import pickle
from tqdm import tqdm
from pathlib import Path
import torch

# 导入必要的模块
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'crysmmnet-main/src'))

from jarvis.core.atoms import Atoms
from graphs import Graph
from tokenizers.normalizers import BertNormalizer


def load_vocab_mappings(vocab_file='crysmmnet-main/src/vocab_mappings.txt'):
    """加载词汇映射"""
    possible_paths = [
        vocab_file,
        './vocab_mappings.txt',
        os.path.join(os.path.dirname(__file__), 'vocab_mappings.txt'),
        os.path.join(os.path.dirname(__file__), 'crysmmnet-main/src/vocab_mappings.txt'),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                mappings = f.read().strip().split('\n')
            return {m[0]: m[2:] for m in mappings}

    raise FileNotFoundError(f"无法找到 vocab_mappings.txt。尝试过的路径: {possible_paths}")


def normalize_text(text, mappings):
    """规范化文本"""
    norm = BertNormalizer(lowercase=False, strip_accents=True,
                         clean_text=True, handle_chinese_chars=True)
    text = norm.normalize_str(text)
    for k, v in mappings.items():
        text = text.replace(k, v)
    return text


def preprocess_dataset(dataset, property_name, output_dir='preprocessed_data'):
    """预处理整个数据集"""

    print(f"\n{'='*80}")
    print(f"预处理数据集: {dataset} - {property_name}")
    print(f"{'='*80}\n")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 数据集路径映射
    dataset_paths = {
        'jarvis': {
            'cif_dir': 'crysmmnet-main/data/cif_files',
            'id_prop_file': 'crysmmnet-main/data/id_prop.csv'
        },
        'matbench': {
            'cif_dir': 'crysmmnet-main/data/megnet_cif_files',
            'id_prop_file': 'crysmmnet-main/data/megnet_id_prop.csv'
        }
    }

    # 数据集名称映射
    dataset_name_mapping = {
        'jarvis': 'jarvis',
        'dft_3d': 'jarvis',
        'matbench': 'megnet',
        'megnet': 'megnet',
    }

    actual_dataset = dataset_name_mapping.get(dataset.lower(), dataset.lower())

    if actual_dataset not in dataset_paths:
        raise ValueError(f"不支持的数据集: {dataset}")

    paths = dataset_paths[actual_dataset]
    cif_dir = paths['cif_dir']
    id_prop_file = paths['id_prop_file']

    print(f"CIF 目录: {cif_dir}")
    print(f"描述文件: {id_prop_file}")

    # 加载词汇映射
    print("\n加载词汇映射...")
    mappings = load_vocab_mappings()
    print(f"✓ 加载了 {len(mappings)} 个映射规则")

    # 读取数据
    print(f"\n读取数据文件...")
    with open(id_prop_file, 'r') as f:
        reader = csv.reader(f)
        headings = next(reader)
        data = [row for row in reader]

    print(f"✓ 总样本数: {len(data)}")

    # 数据集划分（与训练脚本保持一致）
    n_train = int(0.8 * len(data))
    n_val = int(0.1 * len(data))
    n_test = len(data) - n_train - n_val

    splits = {
        'train': data[:n_train],
        'val': data[n_train:n_train + n_val],
        'test': data[n_train + n_val:]
    }

    print(f"\n数据集划分:")
    print(f"  训练集: {n_train}")
    print(f"  验证集: {n_val}")
    print(f"  测试集: {n_test}")

    # 处理每个划分
    for split_name, split_data in splits.items():
        print(f"\n{'='*80}")
        print(f"处理 {split_name} 集...")
        print(f"{'='*80}\n")

        processed_samples = []
        skipped = 0

        for row in tqdm(split_data, desc=f"加载 {split_name}"):
            material_id = row[0]
            text_description = row[1]
            target = float(row[2])

            # CIF 文件路径
            cif_path = os.path.join(cif_dir, f"{material_id}.cif")

            if not os.path.exists(cif_path):
                skipped += 1
                continue

            try:
                # 加载结构
                atoms = Atoms.from_cif(cif_path)

                # 构建图
                graph = Graph.atom_dgl_multigraph(
                    atoms,
                    cutoff=8.0,
                    atom_features="cgcnn",
                    max_neighbors=12,
                    compute_line_graph=True,
                    use_canonize=True,
                )

                # 规范化文本
                normalized_text = normalize_text(text_description, mappings)

                # 保存样本
                sample = {
                    'id': material_id,
                    'graph': graph,
                    'line_graph': graph[1],  # line graph
                    'text': normalized_text,
                    'target': target
                }

                processed_samples.append(sample)

            except Exception as e:
                print(f"\n警告: 无法处理 {material_id}: {e}")
                skipped += 1
                continue

        # 保存预处理数据
        output_file = os.path.join(
            output_dir,
            f"{actual_dataset}_{property_name}_{split_name}.pkl"
        )

        print(f"\n保存预处理数据到: {output_file}")
        with open(output_file, 'wb') as f:
            pickle.dump(processed_samples, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"✓ 保存了 {len(processed_samples)} 个样本")
        print(f"✓ 跳过了 {skipped} 个样本")

        # 显示文件大小
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        print(f"✓ 文件大小: {file_size:.2f} MB")

    print(f"\n{'='*80}")
    print(f"预处理完成！")
    print(f"{'='*80}\n")
    print(f"生成的文件位于: {output_dir}/")
    print(f"  - {actual_dataset}_{property_name}_train.pkl")
    print(f"  - {actual_dataset}_{property_name}_val.pkl")
    print(f"  - {actual_dataset}_{property_name}_test.pkl")
    print(f"\n使用预处理数据训练:")
    print(f"  python train_with_cross_modal_attention.py \\")
    print(f"    --dataset {dataset} \\")
    print(f"    --property {property_name} \\")
    print(f"    --use_preprocessed True")


def main():
    parser = argparse.ArgumentParser(description='预处理图数据')

    parser.add_argument('--dataset', type=str, default='jarvis',
                       choices=['jarvis', 'dft_3d', 'matbench', 'megnet'],
                       help='数据集名称')

    parser.add_argument('--property', type=str, default='mbj_bandgap',
                       help='目标属性名称')

    parser.add_argument('--output_dir', type=str, default='preprocessed_data',
                       help='输出目录')

    args = parser.parse_args()

    preprocess_dataset(
        dataset=args.dataset,
        property_name=args.property,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
