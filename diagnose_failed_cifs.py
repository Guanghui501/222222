#!/usr/bin/env python
"""
诊断无法加载的CIF文件

用于分析哪些CIF文件被所有方法（JARVIS、pymatgen、ase）拒绝，并诊断原因
"""

import sys
import argparse
from pathlib import Path
from tqdm import tqdm

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


def test_jarvis(cif_path):
    """测试JARVIS是否能加载"""
    try:
        from jarvis.core.atoms import Atoms
        import io
        import contextlib

        stderr_buffer = io.StringIO()
        with contextlib.redirect_stderr(stderr_buffer):
            atoms = Atoms.from_cif(str(cif_path))

        composition = atoms.composition.reduced_formula
        return True, f"OK: {composition}"
    except Exception as e:
        return False, str(e)[:100]


def test_pymatgen(cif_path):
    """测试pymatgen是否能加载"""
    try:
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        s = Structure.from_file(str(cif_path))
        sga = SpacegroupAnalyzer(s, symprec=1e-3)
        sgnum = sga.get_space_group_number()
        composition = s.composition.reduced_formula

        return True, f"OK: {composition}, SG={sgnum}"
    except Exception as e:
        return False, str(e)[:100]


def test_ase(cif_path):
    """测试ASE是否能加载"""
    try:
        from ase.io import read
        import spglib

        atoms = read(str(cif_path))
        composition = atoms.get_chemical_formula()

        cell = (atoms.cell.array, atoms.get_scaled_positions(),
                [a.number for a in atoms])
        dataset = spglib.get_symmetry_dataset(cell, symprec=1e-3)
        sgnum = dataset["number"]

        return True, f"OK: {composition}, SG={sgnum}"
    except Exception as e:
        return False, str(e)[:100]


def diagnose_cif(cif_path):
    """全面诊断单个CIF文件"""
    results = {
        'file': cif_path.name,
        'jarvis': test_jarvis(cif_path),
        'pymatgen': test_pymatgen(cif_path),
        'ase': test_ase(cif_path)
    }

    # 统计成功的方法
    success_count = sum([1 for method in ['jarvis', 'pymatgen', 'ase']
                        if results[method][0]])

    results['success_count'] = success_count
    results['all_failed'] = success_count == 0

    return results


def main():
    parser = argparse.ArgumentParser(description='诊断无法加载的CIF文件')
    parser.add_argument('--cif_dir', type=str, required=True,
                       help='CIF文件目录')
    parser.add_argument('--max_files', type=int, default=None,
                       help='最多测试多少个文件（用于快速测试）')
    parser.add_argument('--show_all', action='store_true',
                       help='显示所有文件的结果（默认只显示失败的）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出详细报告到文件')

    args = parser.parse_args()

    # 收集CIF文件
    cif_dir = Path(args.cif_dir)
    cif_files = list(cif_dir.glob('*.cif'))

    if args.max_files:
        cif_files = cif_files[:args.max_files]

    print(f"\n找到 {len(cif_files)} 个CIF文件")
    print("="*80)

    # 诊断所有文件
    all_results = []
    failed_files = []

    for cif_file in tqdm(cif_files, desc="诊断中"):
        result = diagnose_cif(cif_file)
        all_results.append(result)

        if result['all_failed']:
            failed_files.append(result)

    # 统计
    total = len(all_results)
    all_success = sum(1 for r in all_results if r['success_count'] == 3)
    partial_success = sum(1 for r in all_results if 0 < r['success_count'] < 3)
    all_failed = len(failed_files)

    print(f"\n" + "="*80)
    print("统计结果:")
    print("="*80)
    print(f"总文件数: {total}")
    print(f"✅ 全部方法成功: {all_success} ({all_success/total*100:.1f}%)")
    print(f"⚠️  部分方法成功: {partial_success} ({partial_success/total*100:.1f}%)")
    print(f"❌ 全部方法失败: {all_failed} ({all_failed/total*100:.1f}%)")

    # 方法成功率
    jarvis_success = sum(1 for r in all_results if r['jarvis'][0])
    pymatgen_success = sum(1 for r in all_results if r['pymatgen'][0])
    ase_success = sum(1 for r in all_results if r['ase'][0])

    print(f"\n各方法成功率:")
    print(f"  JARVIS:   {jarvis_success}/{total} ({jarvis_success/total*100:.1f}%)")
    print(f"  pymatgen: {pymatgen_success}/{total} ({pymatgen_success/total*100:.1f}%)")
    print(f"  ASE:      {ase_success}/{total} ({ase_success/total*100:.1f}%)")

    # 显示失败的文件
    if failed_files:
        print(f"\n" + "="*80)
        print(f"全部方法都失败的文件 ({len(failed_files)} 个):")
        print("="*80)

        for i, result in enumerate(failed_files[:20], 1):  # 只显示前20个
            print(f"\n{i}. {result['file']}")
            print(f"   JARVIS:   {result['jarvis'][1]}")
            print(f"   pymatgen: {result['pymatgen'][1]}")
            print(f"   ASE:      {result['ase'][1]}")

        if len(failed_files) > 20:
            print(f"\n... 还有 {len(failed_files)-20} 个失败文件未显示")

    # 显示部分成功的文件
    partial_files = [r for r in all_results if 0 < r['success_count'] < 3]
    if partial_files and args.show_all:
        print(f"\n" + "="*80)
        print(f"部分方法成功的文件 ({len(partial_files)} 个):")
        print("="*80)

        for i, result in enumerate(partial_files[:10], 1):
            print(f"\n{i}. {result['file']}")
            status = []
            for method in ['jarvis', 'pymatgen', 'ase']:
                success, msg = result[method]
                status.append(f"{'✓' if success else '✗'} {method}")
            print(f"   状态: {', '.join(status)}")

            # 显示失败方法的错误
            for method in ['jarvis', 'pymatgen', 'ase']:
                success, msg = result[method]
                if not success:
                    print(f"   {method}: {msg}")

    # 保存详细报告
    if args.output:
        with open(args.output, 'w') as f:
            f.write("CIF文件诊断详细报告\n")
            f.write("="*80 + "\n\n")

            f.write(f"总文件数: {total}\n")
            f.write(f"全部成功: {all_success} ({all_success/total*100:.1f}%)\n")
            f.write(f"部分成功: {partial_success} ({partial_success/total*100:.1f}%)\n")
            f.write(f"全部失败: {all_failed} ({all_failed/total*100:.1f}%)\n\n")

            f.write("="*80 + "\n")
            f.write("全部失败的文件详情:\n")
            f.write("="*80 + "\n\n")

            for result in failed_files:
                f.write(f"文件: {result['file']}\n")
                f.write(f"  JARVIS:   {result['jarvis'][1]}\n")
                f.write(f"  pymatgen: {result['pymatgen'][1]}\n")
                f.write(f"  ASE:      {result['ase'][1]}\n")
                f.write("\n")

        print(f"\n详细报告已保存到: {args.output}")


if __name__ == "__main__":
    main()
