# CrysMMNet 代码改进建议

## 一、融合机制改进 (核心)

### 1. 中期融合 (Middle Fusion)
**当前问题**: 图和文本完全独立编码后才拼接，缺乏层间交互

**改进方案**:
#### 方案A: 多层级融合
```python
class MultiLevelFusion(nn.Module):
    """在GNN的多个层级进行融合"""
    def __init__(self, config):
        super().__init__()
        # 在每个ALIGNN层后添加跨模态注意力
        self.cross_modal_attention = nn.ModuleList([
            CrossModalAttention(hidden_dim=256)
            for _ in range(config.alignn_layers)
        ])

    def forward(self, graph_features, text_features):
        # 在每层GNN更新后与文本特征交互
        for i, alignn_layer in enumerate(self.alignn_layers):
            x, y, z = alignn_layer(g, lg, x, y, z)
            # 跨模态注意力增强
            x = self.cross_modal_attention[i](x, text_features)
        return x
```

#### 方案B: 分层文本特征提取
```python
# 不仅使用CLS token，利用BERT的多层输出
class HierarchicalTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model.config.output_hidden_states = True
        # 提取BERT的第6, 9, 12层
        self.layer_weights = nn.Parameter(torch.ones(3) / 3)

    def forward(self, text):
        outputs = self.text_model(**encodings)
        hidden_states = outputs.hidden_states  # (13 layers)
        # 加权融合多层
        selected = torch.stack([hidden_states[6], hidden_states[9], hidden_states[12]])
        weighted = torch.sum(selected * self.layer_weights.view(-1,1,1,1), dim=0)
        return weighted
```

**位置**: 在 `src/models/alignn.py:328-334` (ALIGNN层循环内部)

**预期效果**: MAE降低 5-10%，更好的模态协同

---

### 2. 注意力融合机制 ⭐⭐⭐

**当前问题**: 简单拼接无法学习模态间的重要性权重

**改进方案**:
#### 方案A: 跨模态注意力 (Cross-Modal Attention)
```python
class CrossModalAttention(nn.Module):
    """图-文本跨模态注意力"""
    def __init__(self, graph_dim=256, text_dim=768, hidden_dim=256):
        super().__init__()
        self.query = nn.Linear(graph_dim, hidden_dim)
        self.key = nn.Linear(text_dim, hidden_dim)
        self.value = nn.Linear(text_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, graph_feat, text_feat):
        # graph_feat: [batch, graph_dim]
        # text_feat: [batch, seq_len, text_dim]
        Q = self.query(graph_feat).unsqueeze(1)  # [batch, 1, hidden]
        K = self.key(text_feat)  # [batch, seq_len, hidden]
        V = self.value(text_feat)

        # Attention scores
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Weighted sum
        context = torch.matmul(attn, V).squeeze(1)  # [batch, hidden]
        return graph_feat + context  # 残差连接
```

#### 方案B: 双向注意力融合
```python
class BidirectionalAttentionFusion(nn.Module):
    """图到文本 + 文本到图的双向注意力"""
    def __init__(self, dim=256):
        super().__init__()
        self.graph_to_text = CrossModalAttention(dim, dim, dim)
        self.text_to_graph = CrossModalAttention(dim, dim, dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, graph_emb, text_emb):
        # 图增强文本
        enhanced_text = self.graph_to_text(text_emb, graph_emb)
        # 文本增强图
        enhanced_graph = self.text_to_graph(graph_emb, text_emb)

        # 门控融合
        concat = torch.cat([enhanced_graph, enhanced_text], dim=-1)
        gate = self.fusion_gate(concat)
        fused = gate * enhanced_graph + (1 - gate) * enhanced_text
        return fused
```

**位置**: 替换 `src/models/alignn.py:341` 的简单拼接

**预期效果**: 提升 8-15% 性能

---

### 3. 门控融合 (Gated Fusion) ⭐⭐

**改进方案**:
```python
class GatedMultimodalFusion(nn.Module):
    """自适应学习图和文本的权重"""
    def __init__(self, graph_dim=64, text_dim=64):
        super().__init__()
        self.graph_fc = nn.Linear(graph_dim, graph_dim)
        self.text_fc = nn.Linear(text_dim, text_dim)

        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(graph_dim + text_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # 2个权重：图和文本
            nn.Softmax(dim=-1)
        )

    def forward(self, graph_emb, text_emb):
        # 计算门控权重
        concat = torch.cat([graph_emb, text_emb], dim=-1)
        weights = self.gate(concat)  # [batch, 2]

        # 加权融合
        graph_weight = weights[:, 0:1]
        text_weight = weights[:, 1:2]

        fused = graph_weight * self.graph_fc(graph_emb) + \
                text_weight * self.text_fc(text_emb)
        return fused
```

**优点**: 不同样本自适应调整模态重要性

---

### 4. 对比学习增强 ⭐⭐⭐

**当前问题**: 缺乏模态对齐的显式监督

**改进方案**:
```python
class ContrastiveLoss(nn.Module):
    """图-文本对比学习损失"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, graph_emb, text_emb):
        # L2归一化
        graph_emb = F.normalize(graph_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)

        # 相似度矩阵
        similarity = torch.matmul(graph_emb, text_emb.T) / self.temperature

        # InfoNCE loss
        labels = torch.arange(graph_emb.size(0)).to(graph_emb.device)
        loss_g2t = F.cross_entropy(similarity, labels)
        loss_t2g = F.cross_entropy(similarity.T, labels)

        return (loss_g2t + loss_t2g) / 2

# 训练时添加
total_loss = regression_loss + alpha * contrastive_loss
```

**位置**: 在 `src/train.py` 的损失函数部分添加

**预期效果**: 提升模态对齐，改善泛化性能

---

## 二、文本编码器改进

### 1. 文本编码器微调 ⭐⭐
**当前问题**: `src/models/alignn.py:312` 使用 `with torch.no_grad()`，文本编码器完全冻结

**改进方案**:
```python
# 方案A: 部分微调 (只微调最后几层)
for param in text_model.parameters():
    param.requires_grad = False
for param in text_model.encoder.layer[-3:].parameters():  # 最后3层
    param.requires_grad = True

# 方案B: 使用更小的学习率
text_optimizer = torch.optim.AdamW(
    text_model.parameters(),
    lr=1e-5  # 比图编码器小10倍
)
```

### 2. 多粒度文本特征 ⭐⭐
**改进**: 不仅使用CLS，还使用token级别特征
```python
# 当前只用CLS: cls_emb = last_hidden_state[:, 0, :]
# 改进: 同时使用mean pooling
mean_emb = torch.mean(last_hidden_state[:, 1:-1, :], dim=1)  # 排除CLS和SEP
max_emb = torch.max(last_hidden_state[:, 1:-1, :], dim=1)[0]

# 多特征融合
text_emb = self.text_projection(torch.cat([cls_emb, mean_emb, max_emb], dim=-1))
```

---

## 三、图编码器改进

### 1. 图池化改进 ⭐⭐
**当前问题**: `src/models/alignn.py:267` 只使用AvgPooling

**改进方案**:
```python
class HybridPooling(nn.Module):
    """结合多种池化方式"""
    def __init__(self):
        super().__init__()
        self.avg_pool = AvgPooling()
        self.max_pool = MaxPooling()
        self.attention_pool = GlobalAttentionPooling(
            nn.Linear(hidden_features, 1)
        )
        self.fusion = nn.Linear(hidden_features * 3, hidden_features)

    def forward(self, g, x):
        avg = self.avg_pool(g, x)
        max_p = self.max_pool(g, x)
        attn = self.attention_pool(g, x)
        return self.fusion(torch.cat([avg, max_p, attn], dim=-1))
```

### 2. 图结构增强 ⭐
**改进**: 添加全局虚拟节点
```python
# 在graphs.py中添加虚拟节点连接所有原子
def add_virtual_node(g):
    num_nodes = g.num_nodes()
    g.add_nodes(1)  # 虚拟节点
    # 虚拟节点连接所有节点
    src = [num_nodes] * num_nodes + list(range(num_nodes))
    dst = list(range(num_nodes)) + [num_nodes] * num_nodes
    g.add_edges(src, dst)
    return g
```

---

## 四、训练策略改进

### 1. 课程学习 (Curriculum Learning) ⭐⭐
```python
# 先训练简单样本，逐步增加难度
class CurriculumScheduler:
    def __init__(self, dataset, difficulty_metric='target_variance'):
        # 根据目标值方差排序样本
        self.sorted_indices = self.sort_by_difficulty(dataset)

    def get_curriculum_subset(self, epoch, total_epochs):
        # 前期训练简单样本，后期加入困难样本
        ratio = min(1.0, (epoch + 1) / (total_epochs * 0.5))
        n_samples = int(len(self.sorted_indices) * ratio)
        return self.sorted_indices[:n_samples]
```

### 2. 多任务学习 ⭐⭐
```python
# 同时预测多个相关性质
class MultiTaskHead(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.shared = nn.Linear(hidden_dim, hidden_dim)
        self.task_heads = nn.ModuleDict({
            'formation_energy': nn.Linear(hidden_dim, 1),
            'bandgap': nn.Linear(hidden_dim, 1),
            'bulk_modulus': nn.Linear(hidden_dim, 1)
        })

    def forward(self, x, task_name):
        shared_feat = self.shared(x)
        return self.task_heads[task_name](shared_feat)
```

### 3. 数据增强 ⭐
```python
# 图数据增强
def augment_graph(g):
    # 1. 节点特征加噪声
    g.ndata['atom_features'] += torch.randn_like(g.ndata['atom_features']) * 0.1

    # 2. 边dropout
    edge_mask = torch.rand(g.num_edges()) > 0.1
    g = dgl.edge_subgraph(g, edge_mask, preserve_nodes=True)

    # 3. 子图采样
    sampled_nodes = torch.randperm(g.num_nodes())[:int(g.num_nodes()*0.8)]
    g = dgl.node_subgraph(g, sampled_nodes)

    return g
```

---

## 五、架构层面改进

### 1. Transformer-based GNN ⭐⭐
```python
class GraphTransformerLayer(nn.Module):
    """使用Transformer替代GNN层"""
    def __init__(self, hidden_dim=256, num_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, g, node_feat):
        # 构建节点位置编码
        pos_enc = self.get_graph_positional_encoding(g)
        x = node_feat + pos_enc

        # Self-attention
        attn_out, _ = self.multihead_attn(x, x, x)
        x = self.norm1(x + attn_out)

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x
```

### 2. 图-文本联合预训练 ⭐⭐⭐
```python
# 在大规模材料数据上预训练
class PretrainedCrysMMNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 掩码图预测任务
        self.mask_node_prediction = nn.Linear(hidden_dim, num_atom_types)
        # 掩码文本预测 (类似BERT MLM)
        self.mask_text_prediction = nn.Linear(768, vocab_size)

    def pretrain_step(self, g, text):
        # 图掩码预测
        masked_g, labels = self.mask_graph_nodes(g)
        node_pred = self.mask_node_prediction(masked_g)
        loss_graph = F.cross_entropy(node_pred, labels)

        # 文本掩码预测
        masked_text, text_labels = self.mask_text_tokens(text)
        text_pred = self.mask_text_prediction(masked_text)
        loss_text = F.cross_entropy(text_pred, text_labels)

        # 图-文本匹配任务
        loss_match = self.contrastive_loss(graph_emb, text_emb)

        return loss_graph + loss_text + loss_match
```

---

## 六、实现优先级建议

### 高优先级 (立即实现)
1. ⭐⭐⭐ 跨模态注意力融合
2. ⭐⭐⭐ 中期多层级融合
3. ⭐⭐⭐ 对比学习损失

### 中优先级 (后续优化)
4. ⭐⭐ 门控融合机制
5. ⭐⭐ 文本编码器部分微调
6. ⭐⭐ 混合图池化

### 低优先级 (探索性研究)
7. ⭐ 图数据增强
8. ⭐ Transformer-based GNN
9. ⭐ 多任务学习

---

## 七、代码修改位置总结

| 改进项 | 修改文件 | 修改位置 | 预期提升 |
|--------|---------|---------|---------|
| 中期融合 | `src/models/alignn.py` | 第328-334行循环内 | 5-10% MAE |
| 跨模态注意力 | `src/models/alignn.py` | 第341行替换cat | 8-15% MAE |
| 对比学习 | `src/train.py` | 损失函数部分 | 提升泛化 |
| 文本微调 | `src/models/alignn.py` | 第312行去除no_grad | 3-5% MAE |
| 混合池化 | `src/models/alignn.py` | 第267行替换 | 2-5% MAE |

---

## 八、参考文献

1. **跨模态注意力**: "ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations" (NeurIPS 2019)
2. **对比学习**: "CLIP: Learning Transferable Visual Models From Natural Language Supervision" (ICML 2021)
3. **中期融合**: "LXMERT: Learning Cross-Modality Encoder Representations from Transformers" (EMNLP 2019)
4. **图Transformer**: "Graphormer: Do Transformers Really Perform Bad for Graph Representation?" (NeurIPS 2021)
