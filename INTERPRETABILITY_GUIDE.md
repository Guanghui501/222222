# CrysMMNet 可解释性分析指南

## 📖 概述

CrysMMNet 的可解释性功能帮助您理解模型的预测过程，包括：

1. **跨模态注意力可视化** - 图和文本之间如何相互影响
2. **原子重要性分析** - 哪些原子对预测贡献最大
3. **文本特征重要性** - 文本描述的哪些部分最关键
4. **特征空间可视化** - 图-文本对齐质量分析

---

## 🎯 核心功能

### 1. 跨模态注意力权重

显示模型如何将晶体结构和文本描述关联起来。

**特点**：
- Graph-to-Text 注意力：图特征如何关注文本
- Text-to-Graph 注意力：文本如何关注图特征
- 多头注意力可视化

**用途**：
- 理解模型如何融合两种模态
- 发现哪些描述词与结构特征对应
- 验证跨模态对齐的有效性

---

### 2. 原子重要性分数

计算每个原子对最终预测的贡献度。

**方法**：
- **梯度方法**（快速）：基于梯度的L2范数
- **积分梯度**（更准确）：沿插值路径积分梯度

**可视化**：
- 原子重要性分布直方图
- 按元素类型的平均重要性
- 3D结构投影（标注重要原子）

**应用**：
- 识别活性位点
- 理解结构-性质关系
- 验证化学直觉

---

### 3. 特征空间可视化

使用 t-SNE 或 PCA 将高维特征投影到 2D 空间。

**可视化内容**：
- 图特征和文本特征的分布
- 配对样本的对齐情况
- 根据目标值着色

**用途**：
- 评估对比学习效果
- 检查特征分布
- 发现聚类模式

---

## 🚀 快速开始

### 步骤1: 训练带有可解释性支持的模型

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property mbj_bandgap \
    --use_cross_modal True \
    --use_contrastive True \
    --epochs 200 \
    --output_dir ./output/interpretable_model/
```

**注意**: 只有启用了 `--use_cross_modal True` 的模型才能进行注意力可视化。

---

### 步骤2: 分析单个样本

```bash
python explain_predictions.py \
    --model_path ./output/interpretable_model/best_model.pt \
    --cif_file ./data/sample.cif \
    --text "cubic perovskite structure with high symmetry" \
    --true_value 2.5 \
    --output_dir ./interpretability_results/
```

**生成的文件**：
```
interpretability_results/
└── custom_sample/
    ├── atom_importance.png           # 原子重要性可视化
    ├── attention_graph_to_text.png   # Graph→Text 注意力
    ├── attention_text_to_graph.png   # Text→Graph 注意力
    └── explanation.json               # 详细解释数据
```

---

### 步骤3: 批量分析测试集

```python
from interpretability import InterpretabilityAnalyzer, create_interpretability_report

# 加载模型
model = load_your_model()
analyzer = InterpretabilityAnalyzer(model, device='cuda')

# 生成报告
create_interpretability_report(
    analyzer,
    test_loader,
    save_dir='./reports/',
    num_samples=20
)
```

---

## 📊 解释输出示例

### 1. 原子重要性输出

```
Top 10 Most Important Atoms:
============================================================
Atom    X        Y        Z     Importance
O    2.345    1.234    0.567    0.9823
Fe   0.000    0.000    0.000    0.8954
O    2.345   -1.234    0.567    0.8721
...
============================================================
```

### 2. explanation.json 结构

```json
{
  "sample_id": "JVASP-1234",
  "formula": "Fe2O3",
  "num_atoms": 10,
  "prediction": 2.456,
  "true_value": 2.500,
  "error": 0.044,
  "atom_importance": [0.982, 0.895, ...],
  "text_description": "hematite structure with ..."
}
```

---

## 🔬 进阶用法

### 自定义原子重要性分析

```python
from interpretability import InterpretabilityAnalyzer

# 创建分析器
analyzer = InterpretabilityAnalyzer(model, device='cuda')

# 方法1: 快速梯度方法
importance_grad = analyzer.compute_atom_importance(
    graph,
    method='gradient'
)

# 方法2: 更准确的积分梯度（但更慢）
importance_ig = analyzer.compute_atom_importance(
    graph,
    method='integrated_gradients',
    steps=100  # 积分步数
)

# 可视化比较
analyzer.visualize_atom_importance(
    atoms_object,
    importance_grad,
    save_path='gradient_importance.png'
)
```

---

### 提取注意力权重

```python
# 启用注意力权重返回
output = model(graph_input, return_attention=True)

# 提取权重
attention_weights = output['attention_weights']
g2t_attention = attention_weights['graph_to_text']  # [batch, heads, seq, seq]
t2g_attention = attention_weights['text_to_graph']  # [batch, heads, seq, seq]

# 平均所有头
g2t_avg = g2t_attention.mean(dim=1)

# 自定义可视化
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(g2t_avg[0].cpu().numpy(), cmap='YlOrRd')
plt.title('Graph-to-Text Attention')
plt.xlabel('Text Features')
plt.ylabel('Graph Features')
plt.savefig('custom_attention.png')
```

---

### 特征空间可视化

```python
# 收集特征
graph_features_list = []
text_features_list = []
labels_list = []

for batch in test_loader:
    g, lg, text, labels = batch
    output = model((g, lg, text), return_features=True)

    graph_features_list.append(output['graph_features'].cpu().numpy())
    text_features_list.append(output['text_features'].cpu().numpy())
    labels_list.append(labels.cpu().numpy())

# 合并
graph_features = np.vstack(graph_features_list)
text_features = np.vstack(text_features_list)
labels = np.concatenate(labels_list)

# 可视化
analyzer.visualize_feature_space(
    graph_features,
    text_features,
    labels,
    method='tsne',  # 或 'pca'
    save_path='feature_space.png'
)
```

---

## 📈 可视化图例

### 原子重要性图

<img src="https://via.placeholder.com/800x300?text=Atom+Importance+Visualization" alt="原子重要性" />

- **左图**: 所有原子的重要性分布
- **中图**: 按元素类型的平均重要性
- **右图**: 3D结构投影（颜色表示重要性）

### 注意力热图

<img src="https://via.placeholder.com/800x400?text=Attention+Heatmap" alt="注意力热图" />

- **横轴**: 文本特征/token
- **纵轴**: 原子/图特征
- **颜色**: 注意力权重（红色=高关注度）

### 特征空间图

<img src="https://via.placeholder.com/800x400?text=Feature+Space+Visualization" alt="特征空间" />

- **蓝色圆点**: 图特征
- **红色三角**: 文本特征
- **灰线**: 配对样本的连线
- **颜色渐变**: 目标值大小

---

## 🎓 应用场景

### 1. 模型调试

**问题**: 为什么模型在某些样本上表现差？

**分析**：
```bash
# 找出误差最大的样本
python explain_predictions.py \
    --model_path ./best_model.pt \
    --cif_file ./worst_sample.cif \
    --text "sample description" \
    --true_value 5.0 \
    --output_dir ./debug/
```

**检查**：
- 原子重要性是否合理？
- 注意力是否聚焦在正确的特征？
- 文本描述是否与结构匹配？

---

### 2. 科学发现

**问题**: 哪些结构特征影响带隙？

**分析步骤**：
1. 分析多个高带隙材料
2. 统计高重要性原子的共同特征
3. 识别关键子结构

**示例代码**：
```python
# 分析多个样本
high_bandgap_samples = [...]
importance_scores = []

for sample in high_bandgap_samples:
    importance = analyzer.compute_atom_importance(sample['graph'])
    importance_scores.append(importance)

# 统计分析
avg_importance_by_element = {}
for i, sample in enumerate(high_bandgap_samples):
    for j, atom in enumerate(sample['atoms'].elements):
        if atom not in avg_importance_by_element:
            avg_importance_by_element[atom] = []
        avg_importance_by_element[atom].append(importance_scores[i][j])

# 打印结果
for element, scores in avg_importance_by_element.items():
    print(f"{element}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

---

### 3. 模型对比

**问题**: 对比学习是否改善了特征对齐？

**分析**：
```python
# 加载两个模型
model_baseline = load_model('baseline.pt')
model_contrastive = load_model('with_contrastive.pt')

analyzer_baseline = InterpretabilityAnalyzer(model_baseline)
analyzer_contrastive = InterpretabilityAnalyzer(model_contrastive)

# 可视化特征空间
analyzer_baseline.visualize_feature_space(
    graph_features, text_features, labels,
    save_path='baseline_features.png'
)

analyzer_contrastive.visualize_feature_space(
    graph_features, text_features, labels,
    save_path='contrastive_features.png'
)

# 比较：对比学习应该产生更紧密的图-文本配对
```

---

### 4. 文献支持

**在论文中使用可解释性分析**：

```markdown
我们使用梯度基础的原子重要性分析识别了对带隙预测最关键的原子。
如图X所示，氧原子在高带隙材料中表现出更高的重要性分数...

跨模态注意力可视化（图Y）表明，模型成功地将"立方对称"等描述
与晶体结构的对称性特征关联起来...
```

---

## ⚙️ 性能优化

### 计算成本

| 方法 | 单样本时间 | 内存占用 | 准确度 |
|------|----------|---------|--------|
| 梯度方法 | ~0.1秒 | 低 | 较好 |
| 积分梯度 (50步) | ~5秒 | 中等 | 很好 |
| 积分梯度 (100步) | ~10秒 | 中等 | 最好 |

**建议**：
- 快速分析：使用梯度方法
- 重要样本：使用积分梯度（50-100步）
- 大规模分析：并行化处理

---

### 批处理优化

```python
# 批量计算原子重要性
import concurrent.futures

def analyze_sample(sample_data):
    graph, atoms, text, true_value = sample_data
    importance = analyzer.compute_atom_importance(graph)
    return importance

# 并行处理
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(analyze_sample, samples))
```

---

## 🐛 故障排除

### 问题1: 注意力权重为 None

**原因**: 模型未启用跨模态注意力

**解决方案**:
```bash
# 确保训练时使用了 --use_cross_modal True
python train_with_cross_modal_attention.py --use_cross_modal True ...

# 检查模型配置
python -c "import torch; print(torch.load('model.pt')['config']['use_cross_modal_attention'])"
```

---

### 问题2: 内存不足

**原因**: 积分梯度需要多次前向/后向传播

**解决方案**:
```python
# 减少积分步数
importance = analyzer.compute_atom_importance(
    graph,
    method='integrated_gradients',
    steps=20  # 从 50 降低到 20
)

# 或使用梯度方法
importance = analyzer.compute_atom_importance(graph, method='gradient')
```

---

### 问题3: 可视化不清晰

**原因**: 图像分辨率或配色方案问题

**解决方案**:
```python
# 自定义可视化参数
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300  # 提高分辨率
plt.rcParams['figure.figsize'] = (15, 10)  # 增大图像尺寸

# 或手动调整
analyzer.visualize_atom_importance(
    atoms,
    importance,
    save_path='high_res.png',
    top_k=20  # 显示更多top原子
)
```

---

## 📚 API 参考

### InterpretabilityAnalyzer

```python
class InterpretabilityAnalyzer:
    def __init__(self, model, device='cuda'):
        """初始化分析器"""

    def compute_atom_importance(self, graph, target_output=None, method='gradient'):
        """计算原子重要性

        Returns:
            importance: np.array [num_atoms]
        """

    def visualize_atom_importance(self, atoms_object, importance_scores,
                                  save_path=None, top_k=10):
        """可视化原子重要性

        Returns:
            df: pandas.DataFrame with atom info and importance
        """

    def visualize_cross_modal_attention(self, graph, attention_weights,
                                       atom_symbols=None, text_tokens=None,
                                       save_path=None):
        """可视化跨模态注意力"""

    def visualize_feature_space(self, graph_features, text_features,
                               labels=None, method='tsne', save_path=None):
        """可视化特征空间"""
```

---

## 🎉 总结

CrysMMNet 的可解释性工具提供了多层次的模型理解：

1. **原子级别** - 哪些原子最重要
2. **模态级别** - 图和文本如何交互
3. **特征级别** - 嵌入空间的结构

**最佳实践**：
- ✅ 总是检查高误差样本的解释
- ✅ 使用可视化验证模型行为
- ✅ 将可解释性分析纳入模型开发循环
- ✅ 在论文中报告关键发现

**下一步**：
- 训练一个带跨模态注意力的模型
- 选择几个代表性样本进行分析
- 生成可视化结果
- 总结发现的模式

祝您探索顺利！🚀
