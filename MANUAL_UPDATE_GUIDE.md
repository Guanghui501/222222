# 手动更新指南 - alignn.py

## 🎯 目标

将您本地的 `models/alignn.py` 文件更新为支持跨模态注意力的版本。

---

## 🚀 方法1: 使用自动脚本（推荐）

```bash
cd /public/home/ghzhang/crysmmnet-main/src

# 下载更新脚本（如果还没有）
# 如果您能从其他机器访问GitHub，可以下载这个脚本并上传到服务器

# 运行更新脚本
bash update_alignn.sh
```

---

## 📥 方法2: 手动下载并替换（如果服务器有网络）

### 步骤1: 备份原文件

```bash
cd /public/home/ghzhang/crysmmnet-main/src
cp models/alignn.py models/alignn.py.backup
```

### 步骤2: 下载新文件

```bash
# 使用 curl
curl -o models/alignn.py \
  https://raw.githubusercontent.com/Guanghui501/222222/claude/parse-functionality-011CV5SzJtxst49fCeaEgG5G/crysmmnet-main/src/models/alignn.py

# 或使用 wget
wget -O models/alignn.py \
  https://raw.githubusercontent.com/Guanghui501/222222/claude/parse-functionality-011CV5SzJtxst49fCeaEgG5G/crysmmnet-main/src/models/alignn.py
```

### 步骤3: 验证

```bash
# 检查新类是否存在
grep -n "class CrossModalAttention" models/alignn.py

# 应该输出: 69:class CrossModalAttention(nn.Module):
```

---

## 💻 方法3: 从本地Git仓库复制（如果您有访问权限）

如果您可以访问包含更新代码的本地副本（例如，您克隆了GitHub仓库到另一台机器）：

### 步骤1: 在有网络的机器上克隆

```bash
# 在您的本地机器或有网络访问的机器上
git clone https://github.com/Guanghui501/222222.git
cd 222222
git checkout claude/parse-functionality-011CV5SzJtxst49fCeaEgG5G
```

### 步骤2: 找到文件

```bash
# 文件位置
ls -la crysmmnet-main/src/models/alignn.py
```

### 步骤3: 上传到服务器

使用 scp、rsync 或您喜欢的方式上传：

```bash
# 从本地机器
scp crysmmnet-main/src/models/alignn.py \
  ghzhang@your-server:/public/home/ghzhang/crysmmnet-main/src/models/alignn.py
```

---

## 🔧 方法4: 使用Git（如果服务器是Git仓库）

如果您的 `/public/home/ghzhang/crysmmnet-main` 是一个Git仓库：

```bash
cd /public/home/ghzhang/crysmmnet-main

# 添加我们的远程仓库
git remote add updates https://github.com/Guanghui501/222222.git

# 获取更新
git fetch updates claude/parse-functionality-011CV5SzJtxst49fCeaEgG5G

# 只更新 alignn.py 文件
git checkout updates/claude/parse-functionality-011CV5SzJtxst49fCeaEgG5G -- src/models/alignn.py
```

---

## ✅ 验证更新

更新后，运行以下命令验证：

```bash
cd /public/home/ghzhang/crysmmnet-main/src

# 检查1: CrossModalAttention 类
python3 -c "
from models.alignn import CrossModalAttention
print('✓ CrossModalAttention 类导入成功')
"

# 检查2: ALIGNNConfig 新参数
python3 -c "
from models.alignn import ALIGNNConfig
config = ALIGNNConfig(
    name='alignn',
    use_cross_modal_attention=True,
    cross_modal_num_heads=4
)
print('✓ ALIGNNConfig 新参数工作正常')
print(f'  use_cross_modal_attention={config.use_cross_modal_attention}')
print(f'  cross_modal_num_heads={config.cross_modal_num_heads}')
"

# 检查3: 训练脚本
python3 -c "
import sys
sys.path.insert(0, '.')
from models.alignn import ALIGNN, ALIGNNConfig
config = ALIGNNConfig(
    name='alignn',
    use_cross_modal_attention=True,
    cross_modal_num_heads=4
)
model = ALIGNN(config)
print('✓ ALIGNN 模型创建成功')
print(f'  use_cross_modal_attention={model.use_cross_modal_attention}')
"
```

如果所有检查都通过，说明更新成功！

---

## 📋 更新内容摘要

新版本的 `alignn.py` 包含：

1. **CrossModalAttention 类** (第69-185行)
   - 双向注意力机制
   - 多头注意力
   - 残差连接和层归一化

2. **ALIGNNConfig 更新** (第203-207行)
   ```python
   use_cross_modal_attention: bool = True
   cross_modal_hidden_dim: int = 256
   cross_modal_num_heads: int = 4
   cross_modal_dropout: float = 0.1
   ```

3. **ALIGNN 模型更新**
   - `__init__` 方法：初始化跨模态注意力模块
   - `forward` 方法：使用跨模态注意力融合

---

## 🆘 如果所有方法都失败

如果上述所有方法都不可行，请联系我，我可以：

1. 提供完整的 `alignn.py` 文件内容（可以复制粘贴）
2. 创建一个 patch 文件
3. 提供其他替代方案

或者查看 `ALIGNN_PY_COMPLETE.txt`（如果我创建了这个文件），其中包含完整的文件内容。

---

## 📞 需要帮助？

如果遇到问题，请提供：
1. 您的环境信息（操作系统、Python版本）
2. 错误信息
3. 您尝试过的方法

---

**更新后即可正常训练！** 🎉
