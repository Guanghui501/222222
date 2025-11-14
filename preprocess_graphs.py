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


def get_dataset_paths(root_dir, dataset, property_name):
    """根据数据集和性质获取数据路径（与训练脚本保持一致）"""

    if dataset.lower() == 'jarvis':
        # JARVIS-DFT 数据集
        property_map = {
            'formation_energy': 'formation_energy_peratom',
            'fe': 'formation_energy_peratom',
            'total_energy': 'optb88vdw_total_energy',
            'opt_bandgap': 'optb88vdw_bandgap',
            'mbj_bandgap': 'mbj_bandgap',
            'bulk_modulus': 'bulk_modulus_kv',
            'bulk_modulus_kv': 'bulk_modulus_kv',
            'shear_modulus': 'shear_modulus_gv',
            'shear_modulus_gv': 'shear_modulus_gv',
        }

        prop_folder = property_map.get(property_name, property_name)
        cif_dir = os.path.join(root_dir, f'jarvis/{prop_folder}/cif/')
        id_prop_file = os.path.join(root_dir, f'jarvis/{prop_folder}/description.csv')

    elif dataset.lower() == 'mp':
        # Material Project 数据集
        if property_name in ['formation_energy', 'band_gap']:
            cif_dir = os.path.join(root_dir, 'mp_2018_new/')
            id_prop_file = os.path.join(root_dir, 'mp_2018_new/mat_text.csv')
        elif property_name in ['bulk', 'shear', 'bulk_modulus', 'shear_modulus']:
            cif_dir = os.path.join(root_dir, 'mp_2018_small/cif/')
            id_prop_file = os.path.join(root_dir, 'mp_2018_small/description.csv')
        else:
            raise ValueError(f"Unsupported property for MP dataset: {property_name}")

    elif dataset.lower() == 'toy':
        # 玩具数据集（用于测试）
        cif_dir = os.path.join(root_dir, 'toy/cif/')
        id_prop_file = os.path.join(root_dir, 'toy/description.csv')

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return cif_dir, id_prop_file


def preprocess_dataset(dataset, property_name, root_dir='../dataset/', output_dir='preprocessed_data'):
    """预处理整个数据集"""

    print(f"\n{'='*80}")
    print(f"预处理数据集: {dataset} - {property_name}")
    print(f"{'='*80}\n")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 使用与训练脚本一致的路径获取逻辑
    cif_dir, id_prop_file = get_dataset_paths(root_dir, dataset, property_name)

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

        # 保存预处理数据（使用dataset名称，与训练脚本一致）
        output_file = os.path.join(
            output_dir,
            f"{dataset.lower()}_{property_name}_{split_name}.pkl"
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
    print(f"  - {dataset.lower()}_{property_name}_train.pkl")
    print(f"  - {dataset.lower()}_{property_name}_val.pkl")
    print(f"  - {dataset.lower()}_{property_name}_test.pkl")
    print(f"\n使用预处理数据训练:")
    print(f"  python train_with_cross_modal_attention.py \\")
    print(f"    --dataset {dataset} \\")
    print(f"    --property {property_name} \\")
    print(f"    --use_preprocessed True")


def main():
    parser = argparse.ArgumentParser(description='预处理图数据')

    parser.add_argument('--dataset', type=str, default='jarvis',
                       choices=['jarvis', 'mp', 'toy'],
                       help='数据集名称')

    parser.add_argument('--property', type=str, default='mbj_bandgap',
                       help='目标属性名称')

    parser.add_argument('--root_dir', type=str, default='../dataset/',
                       help='数据集根目录（与训练脚本保持一致）')

    parser.add_argument('--output_dir', type=str, default='preprocessed_data',
                       help='输出目录')

    args = parser.parse_args()

    preprocess_dataset(
        dataset=args.dataset,
        property_name=args.property,
        root_dir=args.root_dir,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
