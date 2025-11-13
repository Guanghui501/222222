#!/usr/bin/env python
"""
分析和比较训练结果
用法: python analyze_results.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def find_experiment_dirs(base_dir='output'):
    """查找所有实验目录"""
    experiments = {}

    if not os.path.exists(base_dir):
        print(f"目录不存在: {base_dir}")
        return experiments

    for root, dirs, files in os.walk(base_dir):
        if 'test_metrics.csv' in files:
            exp_name = root.replace(base_dir, '').strip('/')
            experiments[exp_name] = root

    return experiments

def analyze_experiment(exp_dir):
    """分析单个实验结果"""
    results = {}

    # 读取测试结果
    test_file = os.path.join(exp_dir, 'test_metrics.csv')
    if os.path.exists(test_file):
        test_df = pd.read_csv(test_file)
        if len(test_df) > 0:
            results['test_mae'] = test_df['mae'].iloc[0]
            if 'rmse' in test_df.columns:
                results['test_rmse'] = test_df['rmse'].iloc[0]

    # 读取验证结果
    val_file = os.path.join(exp_dir, 'val_metrics.csv')
    if os.path.exists(val_file):
        val_df = pd.read_csv(val_file)
        if len(val_df) > 0:
            results['best_val_mae'] = val_df['mae'].min()
            results['final_val_mae'] = val_df['mae'].iloc[-1]
            results['epochs_trained'] = len(val_df)
            results['best_epoch'] = val_df['mae'].idxmin() + 1

    # 读取训练结果
    train_file = os.path.join(exp_dir, 'train_metrics.csv')
    if os.path.exists(train_file):
        train_df = pd.read_csv(train_file)
        if len(train_df) > 0:
            results['final_train_loss'] = train_df['loss'].iloc[-1]

    return results

def compare_experiments(base_dir='output'):
    """比较所有实验"""
    experiments = find_experiment_dirs(base_dir)

    if not experiments:
        print(f"在 {base_dir} 中没有找到实验结果")
        return

    print("\n" + "="*80)
    print("实验结果对比")
    print("="*80)

    all_results = []
    for name, path in experiments.items():
        results = analyze_experiment(path)
        if results:
            results['name'] = name
            all_results.append(results)

    if not all_results:
        print("没有可用的结果")
        return

    # 转换为DataFrame
    df = pd.DataFrame(all_results)

    # 按 test_mae 排序
    if 'test_mae' in df.columns:
        df = df.sort_values('test_mae')

    # 打印结果表格
    print("\n按 Test MAE 排序：\n")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)

    print(df.to_string(index=False))

    # 与论文对比
    paper_mae = 0.27
    print("\n" + "="*80)
    print("与论文结果对比 (Paper MAE = 0.27)")
    print("="*80)

    if 'test_mae' in df.columns:
        best_exp = df.iloc[0]
        best_mae = best_exp['test_mae']
        gap = best_mae - paper_mae
        gap_pct = (gap / paper_mae) * 100

        print(f"\n最佳结果: {best_exp['name']}")
        print(f"  Test MAE: {best_mae:.4f}")
        print(f"  与论文差距: +{gap:.4f} ({gap_pct:.1f}%)")

        if best_mae <= 0.27:
            print("  ✅ 已达到或超过论文水平！")
        elif best_mae <= 0.29:
            print("  ⚠️ 接近论文水平，继续优化")
        elif best_mae <= 0.31:
            print("  ⚠️ 有明显改善，继续训练或调参")
        else:
            print("  ❌ 仍有较大差距，需要进一步优化")

    # 绘制对比图
    try:
        plot_comparison(df, paper_mae)
    except Exception as e:
        print(f"\n绘图失败: {e}")

    return df

def plot_comparison(df, paper_mae=0.27):
    """绘制对比图"""
    if 'test_mae' not in df.columns or len(df) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 图1: Test MAE 对比
    ax1 = axes[0]
    bars = ax1.barh(range(len(df)), df['test_mae'], color='steelblue')
    ax1.axvline(x=paper_mae, color='red', linestyle='--', label=f'Paper (MAE={paper_mae})')
    ax1.set_yticks(range(len(df)))
    ax1.set_yticklabels([name[:30] for name in df['name']], fontsize=8)
    ax1.set_xlabel('Test MAE')
    ax1.set_title('Test MAE Comparison')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)

    # 在柱状图上标注数值
    for i, (idx, row) in enumerate(df.iterrows()):
        mae = row['test_mae']
        ax1.text(mae + 0.01, i, f'{mae:.3f}', va='center', fontsize=8)

    # 图2: 训练过程（如果有多个实验的验证曲线）
    ax2 = axes[1]

    for idx, row in df.iterrows():
        exp_name = row['name']
        exp_dir = None
        for name, path in find_experiment_dirs().items():
            if name == exp_name:
                exp_dir = path
                break

        if exp_dir:
            val_file = os.path.join(exp_dir, 'val_metrics.csv')
            if os.path.exists(val_file):
                val_df = pd.read_csv(val_file)
                ax2.plot(val_df['epoch'], val_df['mae'],
                        label=exp_name[:20], alpha=0.7)

    ax2.axhline(y=paper_mae, color='red', linestyle='--', label=f'Paper (MAE={paper_mae})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation MAE')
    ax2.set_title('Learning Curves')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiment_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n对比图已保存到: experiment_comparison.png")

def plot_learning_curve(exp_dir, exp_name=''):
    """绘制单个实验的学习曲线"""
    train_file = os.path.join(exp_dir, 'train_metrics.csv')
    val_file = os.path.join(exp_dir, 'val_metrics.csv')

    if not os.path.exists(val_file):
        print(f"找不到验证文件: {val_file}")
        return

    val_df = pd.read_csv(val_file)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss 曲线
    ax1 = axes[0]
    if os.path.exists(train_file):
        train_df = pd.read_csv(train_file)
        ax1.plot(train_df['epoch'], train_df['loss'], label='Train Loss', alpha=0.7)

    ax1.plot(val_df['epoch'], val_df['loss'], label='Val Loss', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss Curves - {exp_name}')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # MAE 曲线
    ax2 = axes[1]
    ax2.plot(val_df['epoch'], val_df['mae'], label='Val MAE', color='green', alpha=0.7)
    ax2.axhline(y=0.27, color='red', linestyle='--', label='Paper MAE (0.27)')

    # 标注最佳点
    best_idx = val_df['mae'].idxmin()
    best_epoch = val_df.loc[best_idx, 'epoch']
    best_mae = val_df.loc[best_idx, 'mae']
    ax2.scatter([best_epoch], [best_mae], color='red', s=100, zorder=5)
    ax2.annotate(f'Best: {best_mae:.4f}\nEpoch {best_epoch}',
                xy=(best_epoch, best_mae),
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.set_title(f'MAE Curve - {exp_name}')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    output_file = os.path.join(exp_dir, 'learning_curve.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"学习曲线已保存到: {output_file}")

if __name__ == '__main__':
    import sys

    # 可以指定要分析的目录
    base_dir = sys.argv[1] if len(sys.argv) > 1 else 'output'

    print("="*80)
    print("CrysMMNet 实验结果分析工具")
    print("="*80)

    # 对比所有实验
    df = compare_experiments(base_dir)

    # 如果只有一个实验，绘制详细的学习曲线
    if df is not None and len(df) == 1:
        exp_name = df.iloc[0]['name']
        experiments = find_experiment_dirs(base_dir)
        for name, path in experiments.items():
            if name == exp_name:
                print(f"\n绘制 {exp_name} 的详细学习曲线...")
                plot_learning_curve(path, exp_name)
                break

    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)
