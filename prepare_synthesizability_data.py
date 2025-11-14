#!/usr/bin/env python
"""
准备晶体可合成性二分类数据集

从两个CIF文件夹读取数据：
- train-zheng: 可合成晶体（正样本，label=1）
- train-fu: 不可合成晶体（负样本，label=0）

生成训练所需的description.csv文件
"""

import os
import csv
import argparse
from pathlib import Path
from jarvis.core.atoms import Atoms
from tqdm import tqdm
import random


def extract_with_pmg(cif_path, symprec=1e-3, max_wy=12):
    """
    使用pymatgen提取晶体结构的详细信息

    Args:
        cif_path: CIF文件路径
        symprec: 对称性精度
        max_wy: 最大Wyckoff位置数量

    Returns:
        dict: 包含空间群、晶格参数、Wyckoff位置的字典
    """
    try:
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    except ImportError:
        return None

    try:
        s = Structure.from_file(str(cif_path))
        sga = SpacegroupAnalyzer(s, symprec=symprec)
        sgnum = int(sga.get_space_group_number())
        crystal_system = sga.get_crystal_system()  # Get crystal system from pymatgen

        lat = s.lattice
        lattice = dict(
            a=float(lat.a),
            b=float(lat.b),
            c=float(lat.c),
            alpha=float(lat.alpha),
            beta=float(lat.beta),
            gamma=float(lat.gamma)
        )

        # Use SymmetrizedStructure groups to get multiplicity
        symm = sga.get_symmetrized_structure()
        dataset = sga.get_symmetry_dataset()
        wy_letters = dataset.get("wyckoffs", [])

        equiv_sites = getattr(symm, "equivalent_sites", [])
        equiv_indices = getattr(symm, "equivalent_indices", None)

        wyckoff_sites = []
        if equiv_indices is not None:
            for group_sites, group_indices in zip(equiv_sites, equiv_indices):
                if len(wyckoff_sites) >= max_wy:
                    break
                rep = group_sites[0]
                mult = int(len(group_indices))
                idx0 = int(group_indices[0])
                letter = wy_letters[idx0] if idx0 < len(wy_letters) else "?"
                f = rep.frac_coords
                wyckoff_sites.append({
                    "element": rep.species_string,
                    "wyckoff": f"{mult}{letter}",
                    "wyckoff_letter": letter,
                    "multiplicity": mult,
                    "frac": [float(f[0]), float(f[1]), float(f[2])]
                })
        else:
            # Fallback for older versions
            for group in equiv_sites[:max_wy]:
                rep = group[0]
                f = rep.frac_coords
                try:
                    idx = min(range(len(s)),
                            key=lambda i: ((s[i].frac_coords - f) ** 2).sum())
                except Exception:
                    idx = 0
                mult = int(len(group))
                letter = wy_letters[idx] if idx < len(wy_letters) else "?"
                wyckoff_sites.append({
                    "element": rep.species_string,
                    "wyckoff": f"{mult}{letter}",
                    "wyckoff_letter": letter,
                    "multiplicity": mult,
                    "frac": [float(f[0]), float(f[1]), float(f[2])]
                })

        return {
            "spacegroup_number": sgnum,
            "crystal_system": crystal_system,
            "lattice": lattice,
            "wyckoff_sites": wyckoff_sites
        }
    except Exception as e:
        # If pymatgen extraction fails, return None
        return None


def generate_crystal_description_enhanced(cif_path, atoms):
    """
    生成增强的晶体描述（使用pymatgen提取的详细信息）

    Args:
        cif_path: CIF文件路径
        atoms: JARVIS Atoms对象（作为备用）

    Returns:
        description: 文本描述字符串
    """
    # 尝试使用pymatgen提取详细信息
    pmg_data = extract_with_pmg(cif_path)

    # 从JARVIS获取基础信息
    composition = atoms.composition.reduced_formula
    spacegroup = atoms.spacegroup()
    num_atoms = atoms.num_atoms

    # 尝试从JARVIS获取晶系，如果失败则使用pymatgen的
    try:
        lattice_system = atoms.lattice.lattice_system
    except AttributeError:
        lattice_system = None

    if pmg_data is not None:
        # 使用pymatgen提取的详细信息
        sgnum = pmg_data["spacegroup_number"]
        crystal_system = pmg_data["crystal_system"]
        lattice = pmg_data["lattice"]
        wyckoff_sites = pmg_data["wyckoff_sites"]

        # 如果JARVIS没有提供晶系，使用pymatgen的
        if lattice_system is None:
            lattice_system = crystal_system

        # 构建详细描述
        description = f"{composition} crystal with {lattice_system} lattice system and space group {spacegroup} (No. {sgnum}). "

        # 添加晶格参数
        a, b, c = lattice['a'], lattice['b'], lattice['c']
        alpha, beta, gamma = lattice['alpha'], lattice['beta'], lattice['gamma']

        if lattice_system == "cubic":
            description += f"Cubic lattice parameter a={a:.3f} Å. "
        elif lattice_system == "tetragonal":
            description += f"Tetragonal lattice: a={a:.3f} Å, c={c:.3f} Å. "
        elif lattice_system == "orthorhombic":
            description += f"Orthorhombic lattice: a={a:.3f} Å, b={b:.3f} Å, c={c:.3f} Å. "
        elif lattice_system == "hexagonal":
            description += f"Hexagonal lattice: a={a:.3f} Å, c={c:.3f} Å. "
        else:
            description += f"Lattice parameters: a={a:.3f} Å, b={b:.3f} Å, c={c:.3f} Å, α={alpha:.1f}°, β={beta:.1f}°, γ={gamma:.1f}°. "

        # 添加Wyckoff位置信息
        if wyckoff_sites:
            description += f"Contains {len(wyckoff_sites)} crystallographic sites: "
            site_descriptions = []
            for site in wyckoff_sites[:6]:  # 最多描述6个位置
                element = site['element']
                wyckoff = site['wyckoff']
                site_descriptions.append(f"{element} at {wyckoff}")
            description += ", ".join(site_descriptions)
            if len(wyckoff_sites) > 6:
                description += f", and {len(wyckoff_sites)-6} more sites"
            description += "."

    else:
        # 回退到简单描述（如果pymatgen提取失败）
        if lattice_system is not None:
            description = (
                f"{composition} crystal structure with {lattice_system} lattice system, "
                f"space group {spacegroup}, containing {num_atoms} atoms per unit cell"
            )
        else:
            # 如果连晶系都获取不到，使用最简单的描述
            description = (
                f"{composition} crystal structure with space group {spacegroup}, "
                f"containing {num_atoms} atoms per unit cell"
            )

    return description


def generate_crystal_description(atoms):
    """
    为晶体生成文本描述（简单版本，保持向后兼容）

    Args:
        atoms: JARVIS Atoms对象

    Returns:
        description: 文本描述字符串
    """
    # 获取晶体信息
    composition = atoms.composition.reduced_formula
    spacegroup = atoms.spacegroup()

    # 尝试获取晶系
    try:
        lattice_system = atoms.lattice.lattice_system
    except AttributeError:
        lattice_system = None

    # 获取元素列表
    elements = list(set(atoms.elements))
    elements_str = '-'.join(sorted(elements))

    # 获取原子数
    num_atoms = atoms.num_atoms

    # 构建描述
    if lattice_system is not None:
        description = (
            f"{composition} crystal structure with {lattice_system} lattice system, "
            f"space group {spacegroup}, containing {num_atoms} atoms per unit cell, "
            f"composed of {elements_str} elements"
        )
    else:
        description = (
            f"{composition} crystal structure with space group {spacegroup}, "
            f"containing {num_atoms} atoms per unit cell, "
            f"composed of {elements_str} elements"
        )

    return description


def prepare_synthesizability_dataset(
    positive_dir,
    negative_dir,
    output_dir='./synthesizability_data',
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_seed=42
):
    """
    准备可合成性分类数据集

    Args:
        positive_dir: 可合成晶体CIF文件夹（label=1）
        negative_dir: 不可合成晶体CIF文件夹（label=0）
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子
    """

    print("\n" + "="*80)
    print("晶体可合成性二分类数据准备")
    print("="*80)

    # 设置随机种子
    random.seed(random_seed)

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cif_output_dir = output_dir / 'cif'
    cif_output_dir.mkdir(exist_ok=True)

    print(f"\n输出目录: {output_dir}")
    print(f"CIF目录: {cif_output_dir}")

    # 收集所有数据
    all_data = []

    # 处理可合成晶体（正样本）
    print(f"\n处理可合成晶体 (正样本, label=1):")
    print(f"  目录: {positive_dir}")

    positive_cifs = list(Path(positive_dir).glob('*.cif'))
    print(f"  找到 {len(positive_cifs)} 个CIF文件")

    for cif_file in tqdm(positive_cifs, desc="  加载正样本"):
        try:
            atoms = Atoms.from_cif(str(cif_file))
            composition = atoms.composition.reduced_formula

            # 使用增强版描述（包含Wyckoff位置信息）
            description = generate_crystal_description_enhanced(cif_file, atoms)

            # 生成唯一ID
            file_id = f"synth_pos_{cif_file.stem}"

            # 复制CIF文件到输出目录
            new_cif_path = cif_output_dir / f"{file_id}.cif"
            import shutil
            shutil.copy(cif_file, new_cif_path)

            all_data.append({
                'id': file_id,
                'composition': composition,
                'label': 1,  # 可合成
                'text': description,
                'original_file': cif_file.name
            })
        except Exception as e:
            print(f"    ⚠️  跳过文件 {cif_file.name}: {e}")
            continue

    print(f"  ✅ 成功加载 {sum(1 for d in all_data if d['label']==1)} 个正样本")

    # 处理不可合成晶体（负样本）
    print(f"\n处理不可合成晶体 (负样本, label=0):")
    print(f"  目录: {negative_dir}")

    negative_cifs = list(Path(negative_dir).glob('*.cif'))
    print(f"  找到 {len(negative_cifs)} 个CIF文件")

    for cif_file in tqdm(negative_cifs, desc="  加载负样本"):
        try:
            atoms = Atoms.from_cif(str(cif_file))
            composition = atoms.composition.reduced_formula

            # 使用增强版描述（包含Wyckoff位置信息）
            description = generate_crystal_description_enhanced(cif_file, atoms)

            # 生成唯一ID
            file_id = f"synth_neg_{cif_file.stem}"

            # 复制CIF文件到输出目录
            new_cif_path = cif_output_dir / f"{file_id}.cif"
            import shutil
            shutil.copy(cif_file, new_cif_path)

            all_data.append({
                'id': file_id,
                'composition': composition,
                'label': 0,  # 不可合成
                'text': description,
                'original_file': cif_file.name
            })
        except Exception as e:
            print(f"    ⚠️  跳过文件 {cif_file.name}: {e}")
            continue

    print(f"  ✅ 成功加载 {sum(1 for d in all_data if d['label']==0)} 个负样本")

    # 统计
    total_samples = len(all_data)
    num_positive = sum(1 for d in all_data if d['label'] == 1)
    num_negative = sum(1 for d in all_data if d['label'] == 0)

    print(f"\n数据集统计:")
    print(f"  总样本数: {total_samples}")
    print(f"  可合成 (label=1): {num_positive} ({num_positive/total_samples*100:.1f}%)")
    print(f"  不可合成 (label=0): {num_negative} ({num_negative/total_samples*100:.1f}%)")

    # 随机打乱
    random.shuffle(all_data)

    # 划分数据集
    n_train = int(total_samples * train_ratio)
    n_val = int(total_samples * val_ratio)
    n_test = total_samples - n_train - n_val

    train_data = all_data[:n_train]
    val_data = all_data[n_train:n_train+n_val]
    test_data = all_data[n_train+n_val:]

    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_data)} 样本 ({train_ratio*100:.0f}%)")
    print(f"    - 可合成: {sum(1 for d in train_data if d['label']==1)}")
    print(f"    - 不可合成: {sum(1 for d in train_data if d['label']==0)}")
    print(f"  验证集: {len(val_data)} 样本 ({val_ratio*100:.0f}%)")
    print(f"    - 可合成: {sum(1 for d in val_data if d['label']==1)}")
    print(f"    - 不可合成: {sum(1 for d in val_data if d['label']==0)}")
    print(f"  测试集: {len(test_data)} 样本 ({test_ratio*100:.0f}%)")
    print(f"    - 可合成: {sum(1 for d in test_data if d['label']==1)}")
    print(f"    - 不可合成: {sum(1 for d in test_data if d['label']==0)}")

    # 保存CSV文件
    csv_file = output_dir / 'description.csv'
    print(f"\n保存数据到: {csv_file}")

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['id', 'composition', 'label', 'text', 'split'])

        # 写入训练集
        for sample in train_data:
            writer.writerow([
                sample['id'],
                sample['composition'],
                sample['label'],
                sample['text'],
                'train'
            ])

        # 写入验证集
        for sample in val_data:
            writer.writerow([
                sample['id'],
                sample['composition'],
                sample['label'],
                sample['text'],
                'val'
            ])

        # 写入测试集
        for sample in test_data:
            writer.writerow([
                sample['id'],
                sample['composition'],
                sample['label'],
                sample['text'],
                'test'
            ])

    print(f"✅ 数据准备完成！")

    # 创建数据集信息文件
    info_file = output_dir / 'dataset_info.txt'
    with open(info_file, 'w') as f:
        f.write("晶体可合成性二分类数据集\n")
        f.write("="*60 + "\n\n")
        f.write(f"正样本目录: {positive_dir}\n")
        f.write(f"负样本目录: {negative_dir}\n\n")
        f.write(f"总样本数: {total_samples}\n")
        f.write(f"  可合成 (label=1): {num_positive} ({num_positive/total_samples*100:.1f}%)\n")
        f.write(f"  不可合成 (label=0): {num_negative} ({num_negative/total_samples*100:.1f}%)\n\n")
        f.write(f"训练集: {len(train_data)} 样本\n")
        f.write(f"验证集: {len(val_data)} 样本\n")
        f.write(f"测试集: {len(test_data)} 样本\n\n")
        f.write(f"随机种子: {random_seed}\n")

    print(f"\n数据集信息已保存到: {info_file}")
    print("\n" + "="*80)
    print("下一步：运行训练脚本")
    print(f"python train_synthesizability.py --data_dir {output_dir}")
    print("="*80 + "\n")

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='准备晶体可合成性二分类数据集')

    parser.add_argument('--positive_dir', type=str, default='./train-zheng',
                        help='可合成晶体CIF文件夹路径（正样本，label=1）')
    parser.add_argument('--negative_dir', type=str, default='./train-fu',
                        help='不可合成晶体CIF文件夹路径（负样本，label=0）')
    parser.add_argument('--output_dir', type=str, default='./synthesizability_data',
                        help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='测试集比例')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='随机种子')

    args = parser.parse_args()

    # 验证输入目录
    if not os.path.exists(args.positive_dir):
        print(f"❌ 错误: 找不到可合成晶体目录: {args.positive_dir}")
        exit(1)

    if not os.path.exists(args.negative_dir):
        print(f"❌ 错误: 找不到不可合成晶体目录: {args.negative_dir}")
        exit(1)

    # 准备数据集
    prepare_synthesizability_dataset(
        positive_dir=args.positive_dir,
        negative_dir=args.negative_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )
