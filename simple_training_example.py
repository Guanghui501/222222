#!/usr/bin/env python
"""
简单训练示例 - 最小化代码展示如何使用跨模态注意力

这个脚本提供了最简洁的训练代码，适合快速上手。
"""

import os
import sys

# 添加src目录到路径
sys.path.insert(0, './crysmmnet-main/src')


def train_with_cross_modal_attention():
    """使用跨模态注意力训练模型 - 最简单的方式"""

    # 1. 导入必要的模块
    from data import get_train_val_loaders
    from train import train_dgl
    from config import TrainingConfig

    # 2. 设置基础参数
    root_dir = './crysmmnet-main/dataset/'
    dataset = 'jarvis'
    property_name = 'formation_energy'

    # 3. 加载配置
    config_dict = {
        "dataset": dataset,
        "target": "target",
        "atom_features": "cgcnn",
        "neighbor_strategy": "k-nearest",
        "id_tag": "jid",
        "random_seed": 123,

        # 数据划分
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,

        # 训练参数
        "epochs": 100,  # 快速测试用100，实际训练用1000
        "batch_size": 64,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "warmup_steps": 2000,
        "criterion": "mse",
        "optimizer": "adamw",
        "scheduler": "onecycle",

        "output_dir": f"./output/{property_name}/",
        "num_workers": 0,
        "cutoff": 8.0,
        "max_neighbors": 12,
        "use_canonize": True,
        "keep_data_order": False,
        "write_checkpoint": True,
        "write_predictions": True,
        "pin_memory": False,
        "save_dataloader": False,
        "progress": True,

        # 模型配置
        "model": {
            "name": "alignn",
            "alignn_layers": 4,
            "gcn_layers": 4,
            "atom_input_features": 92,
            "edge_input_features": 80,
            "triplet_input_features": 40,
            "embedding_features": 64,
            "hidden_features": 256,
            "output_features": 1,

            # ⭐ 跨模态注意力配置 - 关键部分！
            "use_cross_modal_attention": True,  # 启用跨模态注意力
            "cross_modal_hidden_dim": 256,      # 注意力隐藏维度
            "cross_modal_num_heads": 4,         # 注意力头数
            "cross_modal_dropout": 0.1,         # Dropout率

            "link": "identity",
            "classification": False
        }
    }

    # 4. 准备数据集（这里需要您自己的数据加载逻辑）
    # 如果您已有处理好的数据，可以直接使用
    # 否则，使用 train_with_cross_modal_attention.py 中的完整数据加载代码

    print("正在加载数据...")
    # dataset_array = load_your_dataset()  # 您的数据加载函数

    # 5. 创建配置对象
    config = TrainingConfig(**config_dict)

    # 6. 创建数据加载器
    # train_loader, val_loader, test_loader, prepare_batch = get_train_val_loaders(
    #     dataset_array=dataset_array,
    #     target=config.target,
    #     batch_size=config.batch_size,
    #     # ... 其他参数
    # )

    # 7. 开始训练
    print("开始训练...")
    # train_dgl(
    #     config=config,
    #     train_val_test_loaders=[train_loader, val_loader, test_loader, prepare_batch]
    # )

    print("训练完成！")


def train_without_cross_modal_attention():
    """不使用跨模态注意力训练 - 用于对比"""

    # 只需要将 use_cross_modal_attention 设置为 False
    config_dict = {
        # ... 其他配置相同 ...
        "model": {
            "name": "alignn",
            # ... 其他参数 ...

            # ⭐ 禁用跨模态注意力
            "use_cross_modal_attention": False,  # 使用原始的简单拼接方式
        }
    }

    print("使用原始方法训练（无跨模态注意力）...")
    # 其余训练代码相同


def quick_comparison_experiment():
    """快速对比实验：跨模态注意力 vs 原始方法"""

    configs = [
        {
            "name": "原始方法（baseline）",
            "use_cross_modal": False
        },
        {
            "name": "跨模态注意力（2头）",
            "use_cross_modal": True,
            "num_heads": 2
        },
        {
            "name": "跨模态注意力（4头）",
            "use_cross_modal": True,
            "num_heads": 4
        },
        {
            "name": "跨模态注意力（8头）",
            "use_cross_modal": True,
            "num_heads": 8
        }
    ]

    results = {}

    for exp_config in configs:
        print(f"\n{'='*60}")
        print(f"实验: {exp_config['name']}")
        print(f"{'='*60}\n")

        # 构建训练配置
        model_config = {
            "use_cross_modal_attention": exp_config["use_cross_modal"],
        }
        if exp_config["use_cross_modal"]:
            model_config["cross_modal_num_heads"] = exp_config["num_heads"]

        # 训练模型
        # mae = train_and_evaluate(model_config)
        # results[exp_config['name']] = mae

    # 打印结果对比
    print("\n" + "="*60)
    print("实验结果对比")
    print("="*60)
    for name, mae in results.items():
        print(f"{name}: MAE = {mae:.4f}")


# ============================================================================
# 代码内直接使用示例
# ============================================================================

def example_1_basic_usage():
    """示例1: 基础使用"""
    print("示例1: 基础使用跨模态注意力")

    from models.alignn import ALIGNN, ALIGNNConfig

    # 创建配置
    config = ALIGNNConfig(
        name="alignn",
        alignn_layers=4,
        gcn_layers=4,
        hidden_features=256,

        # 启用跨模态注意力
        use_cross_modal_attention=True,
        cross_modal_hidden_dim=256,
        cross_modal_num_heads=4,
        cross_modal_dropout=0.1
    )

    # 创建模型
    model = ALIGNN(config)

    print(f"模型创建成功！")
    print(f"使用跨模态注意力: {model.use_cross_modal_attention}")
    print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


def example_2_compare_models():
    """示例2: 对比有无跨模态注意力的模型"""
    print("\n示例2: 对比模型大小")

    from models.alignn import ALIGNN, ALIGNNConfig

    # 原始模型（无跨模态注意力）
    config_baseline = ALIGNNConfig(
        name="alignn",
        use_cross_modal_attention=False
    )
    model_baseline = ALIGNN(config_baseline)

    # 跨模态注意力模型
    config_cross_modal = ALIGNNConfig(
        name="alignn",
        use_cross_modal_attention=True,
        cross_modal_num_heads=4
    )
    model_cross_modal = ALIGNN(config_cross_modal)

    # 统计参数
    params_baseline = sum(p.numel() for p in model_baseline.parameters())
    params_cross_modal = sum(p.numel() for p in model_cross_modal.parameters())

    print(f"原始模型参数量: {params_baseline:,}")
    print(f"跨模态注意力模型参数量: {params_cross_modal:,}")
    print(f"增加的参数量: {params_cross_modal - params_baseline:,} ({(params_cross_modal/params_baseline - 1)*100:.1f}%)")


def example_3_custom_config():
    """示例3: 自定义配置"""
    print("\n示例3: 自定义跨模态注意力配置")

    from models.alignn import ALIGNN, ALIGNNConfig

    # 轻量级配置（显存有限时）
    config_light = ALIGNNConfig(
        name="alignn",
        hidden_features=128,
        use_cross_modal_attention=True,
        cross_modal_hidden_dim=128,
        cross_modal_num_heads=2,
        cross_modal_dropout=0.1
    )

    # 标准配置（推荐）
    config_standard = ALIGNNConfig(
        name="alignn",
        hidden_features=256,
        use_cross_modal_attention=True,
        cross_modal_hidden_dim=256,
        cross_modal_num_heads=4,
        cross_modal_dropout=0.1
    )

    # 高性能配置（大数据集）
    config_high = ALIGNNConfig(
        name="alignn",
        hidden_features=512,
        use_cross_modal_attention=True,
        cross_modal_hidden_dim=512,
        cross_modal_num_heads=8,
        cross_modal_dropout=0.05
    )

    print("轻量级配置: 创建成功")
    print("标准配置: 创建成功")
    print("高性能配置: 创建成功")


def example_4_inference():
    """示例4: 推理使用"""
    print("\n示例4: 模型推理")

    import torch
    from models.alignn import ALIGNN, ALIGNNConfig

    # 加载模型
    config = ALIGNNConfig(
        name="alignn",
        use_cross_modal_attention=True,
        cross_modal_num_heads=4
    )
    model = ALIGNN(config)

    # 加载训练好的权重
    # checkpoint = torch.load('checkpoint_best.pt')
    # model.load_state_dict(checkpoint['model'])

    model.eval()

    print("模型已切换到评估模式")
    print("准备进行推理...")

    # 推理代码
    # with torch.no_grad():
    #     for batch in test_loader:
    #         predictions = model(batch)
    #         # 处理预测结果


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("CrysMMNet 跨模态注意力 - 简单使用示例")
    print("="*60)

    # 运行示例
    example_1_basic_usage()
    example_2_compare_models()
    example_3_custom_config()
    example_4_inference()

    print("\n" + "="*60)
    print("所有示例运行完成！")
    print("="*60)
    print("\n要开始实际训练，请运行:")
    print("  python train_with_cross_modal_attention.py --help")
    print("\n查看完整参数列表。")
