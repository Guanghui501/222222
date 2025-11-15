#!/usr/bin/env python
"""
清理有问题的CIF文件

修复常见的CIF格式问题，使其能被JARVIS/pymatgen/ASE正确解析
"""

import re
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil


def clean_cif_content(content, filename):
    """
    清理CIF文件内容

    修复:
    1. 移除各向异性温度因子 (anisotropic thermal parameters)
    2. 修复无效的数值字段 (如 density_diffrn = "l")
    3. 移除其他可能导致解析失败的字段

    Args:
        content: CIF文件内容
        filename: 文件名（用于日志）

    Returns:
        cleaned_content: 清理后的内容
        changes: 修改列表
    """
    original_content = content
    changes = []

    # 1. 移除各向异性温度因子块
    # 这些参数很多解析器不支持
    aniso_pattern = r'loop_\s+_atom_site_aniso.*?(?=\n(?:loop_|data_|#|$))'
    if re.search(aniso_pattern, content, re.DOTALL):
        content = re.sub(aniso_pattern, '', content, flags=re.DOTALL)
        changes.append("移除各向异性温度因子")

    # 2. 修复无效的density字段
    # 例如: _exptl_crystal_density_diffrn l -> _exptl_crystal_density_diffrn ?
    invalid_density_pattern = r'(_exptl_crystal_density_diffrn)\s+[a-zA-Z]+'
    if re.search(invalid_density_pattern, content):
        content = re.sub(invalid_density_pattern, r'\1 ?', content)
        changes.append("修复无效的密度值")

    # 3. 修复无效的数值字段（通用）
    # 查找应该是数字但是是字母的情况
    numeric_fields = [
        '_cell_length_a',
        '_cell_length_b',
        '_cell_length_c',
        '_cell_angle_alpha',
        '_cell_angle_beta',
        '_cell_angle_gamma',
        '_cell_volume',
        '_cell_formula_units_Z'
    ]

    for field in numeric_fields:
        # 匹配 field后面跟着非数字字符（除了?和.）
        pattern = rf'({field})\s+([a-zA-Z]+)'
        if re.search(pattern, content):
            content = re.sub(pattern, r'\1 ?', content)
            if f"修复无效的{field}值" not in changes:
                changes.append(f"修复无效的{field}值")

    # 4. 移除可能导致问题的注释和额外信息
    # 有时CIF文件末尾有额外的非标准内容
    # （保守处理，只移除明显的尾部垃圾）

    # 5. 规范化空行
    content = re.sub(r'\n{3,}', '\n\n', content)

    return content, changes


def clean_cif_file(input_path, output_path=None, backup=True):
    """
    清理单个CIF文件

    Args:
        input_path: 输入CIF文件路径
        output_path: 输出路径（None则覆盖原文件）
        backup: 是否备份原文件

    Returns:
        success: 是否成功
        changes: 修改列表
    """
    try:
        # 读取文件
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # 清理内容
        cleaned_content, changes = clean_cif_content(content, input_path.name)

        # 如果没有修改，直接返回
        if not changes:
            return True, []

        # 确定输出路径
        if output_path is None:
            output_path = input_path

            # 创建备份
            if backup:
                backup_path = input_path.parent / f"{input_path.stem}.cif.bak"
                shutil.copy2(input_path, backup_path)

        # 写入清理后的文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)

        return True, changes

    except Exception as e:
        return False, [f"错误: {str(e)}"]


def main():
    parser = argparse.ArgumentParser(
        description='清理CIF文件，修复常见格式问题'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                       help='输入CIF文件目录')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录（默认覆盖原文件）')
    parser.add_argument('--no_backup', action='store_true',
                       help='不创建备份文件')
    parser.add_argument('--max_files', type=int, default=None,
                       help='最多处理多少个文件')
    parser.add_argument('--dry_run', action='store_true',
                       help='仅检查，不实际修改文件')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    # 创建输出目录
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # 收集CIF文件
    cif_files = list(input_dir.glob('*.cif'))
    if args.max_files:
        cif_files = cif_files[:args.max_files]

    print(f"\n找到 {len(cif_files)} 个CIF文件")
    print(f"{'[干运行模式]' if args.dry_run else ''}")
    print("="*80)

    # 统计
    total = 0
    modified = 0
    failed = 0
    change_stats = {}

    # 处理所有文件
    for cif_file in tqdm(cif_files, desc="清理CIF文件"):
        total += 1

        # 确定输出路径
        if output_dir:
            out_path = output_dir / cif_file.name
        else:
            out_path = None

        # 清理文件
        if args.dry_run:
            # 干运行：只检查
            with open(cif_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            _, changes = clean_cif_content(content, cif_file.name)
            success = True
        else:
            # 实际修改
            success, changes = clean_cif_file(
                cif_file,
                out_path,
                backup=not args.no_backup
            )

        if not success:
            failed += 1
        elif changes:
            modified += 1
            # 统计修改类型
            for change in changes:
                change_stats[change] = change_stats.get(change, 0) + 1

    # 输出统计
    print("\n" + "="*80)
    print("处理完成!")
    print("="*80)
    print(f"总文件数: {total}")
    print(f"已修改: {modified} ({modified/total*100:.1f}%)")
    print(f"未修改: {total-modified-failed}")
    print(f"失败: {failed}")

    if change_stats:
        print(f"\n修改类型统计:")
        for change_type, count in sorted(change_stats.items(), key=lambda x: -x[1]):
            print(f"  - {change_type}: {count} 个文件")

    if args.dry_run:
        print(f"\n这是干运行模式，未实际修改文件")
        print(f"移除 --dry_run 参数以实际执行修改")
    elif not args.no_backup:
        print(f"\n原始文件已备份为 *.cif.bak")

    if output_dir:
        print(f"\n清理后的文件保存在: {output_dir}")


if __name__ == "__main__":
    main()
