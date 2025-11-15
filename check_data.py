#!/usr/bin/env python
"""检查数据集文件"""

import sys
import csv
from pathlib import Path
from collections import Counter

if len(sys.argv) < 2:
    print("Usage: python check_data.py <data_dir>")
    sys.exit(1)

data_dir = sys.argv[1]
csv_file = Path(data_dir) / 'description.csv'
cif_dir = Path(data_dir) / 'cif'

print(f"检查数据目录: {data_dir}")
print("="*80)

# 检查CSV文件
if not csv_file.exists():
    print(f"❌ CSV文件不存在: {csv_file}")
    sys.exit(1)

print(f"✅ CSV文件存在: {csv_file}")

# 读取CSV
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"\n总行数: {len(rows)}")

# 检查列
if rows:
    print(f"列: {list(rows[0].keys())}")

# 统计split
splits = Counter(row.get('split', 'MISSING') for row in rows)
print(f"\nSplit分布:")
for split, count in splits.items():
    print(f"  {split}: {count} ({count/len(rows)*100:.1f}%)")

# 统计label
labels = Counter(row.get('label', 'MISSING') for row in rows)
print(f"\nLabel分布:")
for label, count in labels.items():
    print(f"  {label}: {count} ({count/len(rows)*100:.1f}%)")

# 检查CIF目录
if not cif_dir.exists():
    print(f"\n❌ CIF目录不存在: {cif_dir}")
    sys.exit(1)

cif_files = list(cif_dir.glob('*.cif'))
print(f"\n✅ CIF目录存在: {cif_dir}")
print(f"CIF文件数: {len(cif_files)}")

# 检查前几个样本
print(f"\n前5个样本:")
for i, row in enumerate(rows[:5], 1):
    cif_path = cif_dir / f"{row['id']}.cif"
    exists = "✅" if cif_path.exists() else "❌"
    print(f"{i}. {row['id']} - split:{row['split']}, label:{row['label']} - CIF:{exists}")
