# 对比学习（Contrastive Learning）使用指南

## 📖 什么是对比学习损失？

对比学习是一种表示学习方法，旨在让相似的样本在特征空间中靠近，不相似的样本远离。在 CrysMMNet 中，我们使用对比学习来对齐图（graph）和文本（text）两种模态的表示。

### 核心思想

对于每个材料样本，我们有两种表示：
- **图表示**：从晶体结构通过 ALIGNN 网络提取的特征
- **文本表示**：从材料描述通过 MatSciBERT 提取的特征

对比学习的目标是：
- 同一材料的图表示和文本表示应该相似（正样本对）
- 不同材料的图表示和文本表示应该不相似（负样本对）

### InfoNCE 损失函数

我们使用 InfoNCE（Noise Contrastive Estimation）损失：

```
相似度矩阵 = (图特征 · 文本特征^T) / 温度

损失_图到文本 = CrossEntropy(相似度矩阵, 对角线标签)
损失_文本到图 = CrossEntropy(相似度矩阵^T, 对角线标签)

总对比损失 = (损失_图到文本 + 损失_文本到图) / 2
```

**温度参数**（temperature）控制分布的平滑度：
- 较低温度（0.07-0.1）：使模型更关注难区分的负样本
- 较高温度（0.2-0.5）：损失分布更平滑

---

## 🎯 为什么在 CrysMMNet 中使用对比学习？

### 优势

1. **增强多模态对齐**：
   - 显式地鼓励图和文本特征的语义一致性
   - 帮助模型学习更好的跨模态表示

2. **改善特征质量**：
   - 作为辅助任务提供额外的学习信号
   - 防止过拟合于主任务

3. **提升下游性能**：
   - 更好的特征表示通常带来更好的预测性能
   - 特别是在数据量有限时效果明显

### 适用场景

✅ **推荐使用对比学习**：
- 使用跨模态注意力机制时（`--use_cross_modal True`）
- 数据集较小时（<10k 样本）
- 希望增强模态对齐时

⚠️ **谨慎使用**：
- 训练速度要求很高时（对比损失会增加少量计算开销）
- 已经过拟合时（需要调整权重）

❌ **不建议使用**：
- 不使用跨模态注意力时（`--use_cross_modal False`）
- 只使用单一模态时

---

## 🚀 快速开始

### 基本用法

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property mbj_bandgap \
    --use_cross_modal True \
    --use_contrastive True \
    --contrastive_weight 0.1 \
    --contrastive_temperature 0.1 \
    --epochs 200 \
    --batch_size 64
```

### 关键参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_contrastive` | bool | False | 是否启用对比学习损失 |
| `--contrastive_weight` | float | 0.1 | 对比损失的权重（相对于主任务损失） |
| `--contrastive_temperature` | float | 0.1 | 温度参数 |

---

## ⚙️ 参数调优指南

### 1. 对比损失权重（contrastive_weight）

控制对比学习在总损失中的比例：

```
总损失 = 主任务损失 + contrastive_weight × 对比损失
```

**推荐值**：

| 权重 | 场景 | 说明 |
|------|------|------|
| 0.05-0.1 | **默认推荐** | 平衡主任务和对比学习 |
| 0.01-0.05 | 主任务优先 | 轻微增强多模态对齐 |
| 0.1-0.2 | 强对齐需求 | 强调跨模态一致性 |
| 0.2+ | 实验性 | 可能过度强调对比学习 |

**调优策略**：

```bash
# 网格搜索
for weight in 0.05 0.1 0.15 0.2; do
    python train_with_cross_modal_attention.py \
        --use_contrastive True \
        --contrastive_weight $weight \
        --output_dir ./output/contrastive_w${weight}/
done
```

**观察指标**：
- 验证 MAE 是否降低
- 训练/验证损失曲线是否平滑
- 是否出现过拟合（验证损失上升）

---

### 2. 温度参数（contrastive_temperature）

控制相似度分布的锐度：

**推荐值**：

| 温度 | 适用场景 | 特点 |
|------|----------|------|
| 0.07 | 大 batch size（≥128） | 更关注难负样本 |
| **0.1** | **默认推荐**（batch=64） | 平衡难度 |
| 0.2 | 小 batch size（≤32） | 更平滑的损失 |
| 0.5 | 初步探索 | 温和的对比学习 |

**经验法则**：
- batch size 越大 → 温度越低（更多负样本，需要更难的对比）
- batch size 越小 → 温度越高（负样本少，避免过度惩罚）

**调优实验**：

```bash
# 测试不同温度
for temp in 0.05 0.1 0.15 0.2; do
    python train_with_cross_modal_attention.py \
        --use_contrastive True \
        --contrastive_temperature $temp \
        --batch_size 64 \
        --output_dir ./output/contrastive_t${temp}/
done
```

---

## 📊 实验示例

### 示例1: 标准配置

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property mbj_bandgap \
    --use_cross_modal True \
    --use_contrastive True \
    --contrastive_weight 0.1 \
    --contrastive_temperature 0.1 \
    --epochs 200 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --output_dir ./output/with_contrastive/
```

**预期效果**：
- 对比基线（无对比学习）可能降低 MAE 0.01-0.03
- 训练曲线更平滑

---

### 示例2: 强对比学习

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property mbj_bandgap \
    --use_cross_modal True \
    --use_contrastive True \
    --contrastive_weight 0.2 \
    --contrastive_temperature 0.07 \
    --epochs 200 \
    --batch_size 128 \
    --learning_rate 0.0008 \
    --output_dir ./output/strong_contrastive/
```

**适用场景**：
- 数据量较小（<5k 样本）
- 希望最大化多模态对齐

---

### 示例3: 轻量对比学习

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property mbj_bandgap \
    --use_cross_modal True \
    --use_contrastive True \
    --contrastive_weight 0.05 \
    --contrastive_temperature 0.15 \
    --epochs 200 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --output_dir ./output/light_contrastive/
```

**适用场景**：
- 主任务性能已经很好
- 只需轻微增强

---

## 🔍 效果监控

### 1. 检查损失组成

训练输出应显示：

```
对比学习配置:
  启用: True
  损失权重: 0.1
  温度参数: 0.1

Epoch 1/200:
  Train Loss: 0.450 (Task: 0.400, Contrastive: 0.500)
  Val Loss: 0.480, Val MAE: 0.320
```

### 2. 分析训练曲线

使用分析工具：

```bash
python analyze_results.py output/with_contrastive
```

**好的信号**：
- ✅ 验证 MAE 低于基线
- ✅ 训练/验证曲线平滑
- ✅ 无明显过拟合

**问题信号**：
- ❌ 验证损失上升 → 降低 contrastive_weight
- ❌ 训练极慢收敛 → 提高 contrastive_temperature
- ❌ 性能反而下降 → 检查配置，考虑禁用

---

## 🧪 对比实验

### 实验设计

对比三种配置：

```bash
# 1. 基线（无对比学习）
python train_with_cross_modal_attention.py \
    --use_cross_modal True \
    --use_contrastive False \
    --epochs 200 \
    --output_dir ./output/baseline/

# 2. 轻度对比学习
python train_with_cross_modal_attention.py \
    --use_cross_modal True \
    --use_contrastive True \
    --contrastive_weight 0.05 \
    --epochs 200 \
    --output_dir ./output/light_contrast/

# 3. 标准对比学习
python train_with_cross_modal_attention.py \
    --use_cross_modal True \
    --use_contrastive True \
    --contrastive_weight 0.1 \
    --epochs 200 \
    --output_dir ./output/standard_contrast/
```

### 结果比较

```bash
python analyze_results.py output
```

期望看到的改进：

| 配置 | 预期 Test MAE | 改进幅度 |
|------|--------------|---------|
| 基线 | 0.300 | - |
| 轻度对比 | 0.290-0.295 | ↓2-3% |
| 标准对比 | 0.280-0.290 | ↓3-7% |

---

## ⚠️ 常见问题

### Q1: 对比学习会增加多少训练时间？

**A**: 通常增加 5-10% 的训练时间。

- 对比损失计算很高效（仅矩阵乘法）
- 主要开销在前向传播时返回中间特征

### Q2: 什么时候不应该使用对比学习？

**A**: 以下情况不建议使用：

1. 不使用跨模态注意力（`--use_cross_modal False`）
2. 已经严重过拟合
3. 训练时间极度受限

### Q3: 如何知道对比学习是否起作用？

**A**: 检查以下指标：

1. **验证 MAE 降低**: 最直接的证据
2. **损失曲线平滑**: 对比学习有正则化效果
3. **特征质量**: 可视化特征分布（使用 t-SNE）

### Q4: contrastive_weight 和 contrastive_temperature 如何协同调整？

**A**: 推荐策略：

1. **先固定温度**（0.1），调整权重：
   - 尝试 [0.05, 0.1, 0.15]
   - 选择验证 MAE 最低的

2. **再固定权重**，微调温度：
   - 根据 batch size 调整
   - batch=32 → temp=0.15
   - batch=64 → temp=0.1
   - batch=128 → temp=0.07

### Q5: 对比学习能否与中期融合同时使用？

**A**: 可以，推荐配置：

```bash
python train_with_cross_modal_attention.py \
    --use_cross_modal True \
    --use_middle_fusion True \
    --middle_fusion_layers "2" \
    --use_contrastive True \
    --contrastive_weight 0.1 \
    --epochs 200
```

中期融合和对比学习是互补的：
- 中期融合：更早地融合模态信息
- 对比学习：增强最终特征的对齐

---

## 🎓 技术细节

### 实现位置

对比学习的核心实现在 `crysmmnet-main/src/models/alignn.py` 中：

**ContrastiveLoss 类**（lines 69-118）：
```python
class ContrastiveLoss(nn.Module):
    """Contrastive loss for aligning graph and text representations."""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, graph_features, text_features):
        # L2 normalize for cosine similarity
        graph_features = F.normalize(graph_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(graph_features, text_features.T) / self.temperature

        # Bidirectional InfoNCE loss
        labels = torch.arange(batch_size, device=graph_features.device)
        loss_g2t = F.cross_entropy(similarity_matrix, labels)
        loss_t2g = F.cross_entropy(similarity_matrix.T, labels)

        return (loss_g2t + loss_t2g) / 2.0
```

**模型输出修改**（ALIGNN.forward）：
- 训练时：返回字典包含 `{'predictions', 'contrastive_loss'}`
- 推理时：只返回 `predictions`

**损失组合**（contrastive_trainer.py）：
```python
class CombinedLoss(nn.Module):
    def forward(self, output, target):
        if isinstance(output, dict):
            task_loss = self.task_criterion(output['predictions'], target)
            contrastive_loss = output.get('contrastive_loss', 0)
            return task_loss + self.contrastive_weight * contrastive_loss
        return self.task_criterion(output, target)
```

---

## 📚 理论背景

### InfoNCE 损失的数学表达

对于 batch size = N 的批次：

1. **特征归一化**：
   ```
   g_i = graph_features[i] / ||graph_features[i]||
   t_i = text_features[i] / ||text_features[i]||
   ```

2. **相似度矩阵**：
   ```
   S[i,j] = exp(g_i · t_j / τ)
   ```
   其中 τ 是温度参数

3. **对比损失**（图到文本）：
   ```
   L_g2t = -1/N ∑_i log(S[i,i] / ∑_j S[i,j])
   ```

4. **双向损失**：
   ```
   L_contrastive = (L_g2t + L_t2g) / 2
   ```

### 为什么需要温度参数？

温度参数 τ 控制 softmax 分布的"尖锐度"：

- **τ → 0**: 分布极度集中于最大值（过度自信）
- **τ = 1**: 标准 softmax
- **τ → ∞**: 均匀分布（无区分度）

典型值 τ=0.1 使模型在保持区分度的同时避免数值不稳定。

---

## 🔬 推荐阅读

1. **SimCLR** - A Simple Framework for Contrastive Learning of Visual Representations (Chen et al., 2020)
2. **CLIP** - Learning Transferable Visual Models From Natural Language Supervision (Radford et al., 2021)
3. **InfoNCE** - Representation Learning with Contrastive Predictive Coding (Oord et al., 2018)

---

## 📞 需要帮助？

如果使用对比学习后：

✅ **性能改善** → 尝试微调参数进一步优化
⚠️ **效果不明显** → 检查是否启用了跨模态注意力
❌ **性能下降** → 降低 contrastive_weight 或禁用对比学习

**调试清单**：
- [ ] 确认 `--use_cross_modal True`
- [ ] 确认训练输出显示对比学习已启用
- [ ] 检查 batch size 和温度参数是否匹配
- [ ] 对比有/无对比学习的结果

---

## 🚀 快速测试脚本

保存为 `test_contrastive.sh`：

```bash
#!/bin/bash

echo "========================================="
echo "测试对比学习效果"
echo "========================================="

# 基线（无对比学习）
echo "运行基线实验..."
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property mbj_bandgap \
    --use_cross_modal True \
    --use_contrastive False \
    --epochs 100 \
    --batch_size 64 \
    --output_dir ./output/baseline/

# 对比学习
echo "运行对比学习实验..."
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property mbj_bandgap \
    --use_cross_modal True \
    --use_contrastive True \
    --contrastive_weight 0.1 \
    --contrastive_temperature 0.1 \
    --epochs 100 \
    --batch_size 64 \
    --output_dir ./output/with_contrastive/

# 分析结果
echo "========================================="
echo "对比结果："
echo "========================================="
python analyze_results.py output/
```

使用：
```bash
chmod +x test_contrastive.sh
bash test_contrastive.sh
```

---

**祝实验顺利！如果对比学习能够帮助您接近论文的 MAE=0.27，请分享您的成功经验！** 🎉
