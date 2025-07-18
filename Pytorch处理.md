# 🧠 PyTorch 中线性变换（`nn.Linear`）与自注意力机制中的矩阵乘法理解

## 📌 问题背景

在 Transformer 的自注意力机制中，通常会使用以下代码：

```python
self.W_q = nn.Linear(hidden_dim, hidden_dim)
Q = self.W_q(X)  # X shape: [batch_size, seq_len, hidden_dim]
```

初学者常有疑问：
> 输入 `X` 是三维的 `[B, S, D]`，而权重矩阵是二维的 `[D, D]`，它们怎么相乘？PyTorch 是不是自动拆分了 token？

---

## ✅ 答案总结

**PyTorch 不会显式地“拆分” token**，但它会自动识别输入张量的 **最后一个维度为特征维度**，并对该维度执行线性变换。

也就是说：

- 输入：`[batch_size, seq_len, hidden_dim]`
- 权重矩阵：`[hidden_dim, hidden_dim]`
- 输出：`[batch_size, seq_len, hidden_dim]`

这个过程等价于对每个 token 向量分别进行一次线性变换。

---

## 🔍 具体原理

### 1. `nn.Linear` 的工作机制

```python
layer = nn.Linear(in_features, out_features)
output = layer(input)
Q = self.W_q(X)  # X.shape = [batch_size, seq_len, hidden_dim]
```  


Q = self.W_q(X)  # X.shape = [batch_size, seq_len, hidden_dim]等价于下面代码
```python
for i in range(batch_size):
    for j in range(seq_len):
        Q[i, j] = W_q(X[i, j])
```

- `nn.Linear` 始终作用于 **输入张量的最后一个维度**
- 它支持任意维度的输入，不限于二维
- 所以即使输入是三维 `[B, S, D]`，它也会自动对每个 `[D]` 向量做变换

### 2. 矩阵乘法示例

设：
- `X.shape = [1, 3, 4]`（batch=1, seq_len=3, hidden_dim=4）
- `W_q.shape = [4, 4]`

则：

```python
Q = W_q(X)  # Q.shape = [1, 3, 4]
```

相当于对每个 token 向量做了：

```python
Q_i = X_i @ W_q.T + b_q
```

---

## 📈 数学表达

设输入张量：

$$
X \in \mathbb{R}^{B \times S \times D}
$$

权重矩阵：

$$
W_q \in \mathbb{R}^{D \times D}
$$

输出张量：

$$
Q = X W_q^T + b_q \Rightarrow Q \in \mathbb{R}^{B \times S \times D}
$$

---

## 🧪 实现方式对比

### 方式一：直接使用 `nn.Linear`

```python
Q = W_q(X)  # 推荐做法，简洁高效
```

### 方式二：手动 reshape 成二维再计算

```python
B, S, D = X.shape
X_flat = X.view(B * S, D)
Q_flat = W_q(X_flat)
Q = Q_flat.view(B, S, D)
```

两种方式结果一致，性能上基本没有差别。

---

## 🧠 总结一句话

> **PyTorch 在执行 `nn.Linear` 时不会显式拆分 token，而是对输入张量的最后一个维度进行矩阵乘法运算，自动实现对每个 token 的线性变换。**

---

## 💡 拓展学习建议

如果你感兴趣，还可以继续学习以下内容：
- 自注意力机制中 Q/K/V 的具体用途
- 多头注意力（Multi-Head Attention）的工作原理
- 使用 `torch.einsum` 更清晰地表达多维运算
- 如何可视化 token 向量和 attention 分布

---


# 🎯 多头注意力（Multi-Head Attention）简介与简单实现

## 🤔 什么是多头注意力？

多头注意力是 Transformer 的核心组件之一。它的思想是：

- 将输入映射到多个不同的子空间（即多个“头”）
- 每个头独立计算 attention score 和加权求和
- 最后将各个头的结果拼接起来，并通过一个线性层整合

这样做的好处是模型可以从不同角度提取信息，增强表达能力。

---

## 📐 输入输出定义

假设：

- `batch_size = B`
- `seq_len = S`
- `hidden_dim = D`
- `num_heads = H`
- 每个 head 的维度为：`d_k = D // H`

---

## 🧱 简单实现步骤

我们来构建一个多头注意力模块的核心部分（忽略 softmax、masking 等细节）：

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.d_k = hidden_dim // num_heads
        
        # 线性投影层
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X):
        batch_size, seq_len, _ = X.size()
        
        # 线性变换
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        # 分头：[B, S, H, d_k] -> [B, H, S, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算 attention scores（简化版）
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)

        # 合并头：[B, H, S, d_k] -> [B, S, H*d_k]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)

        # 最终线性变换
        output = self.W_o(context)
        
        return output
```

---

## 🧪 示例使用

```python
model = MultiHeadAttention(hidden_dim=768, num_heads=8)
X = torch.randn(2, 10, 768)  # batch_size=2, seq_len=10, hidden_dim=768
output = model(X)
print(output.shape)  # [2, 10, 768]
```

---

## 📊 可视化理解（伪代码）

你可以把整个流程想象成：

```
Input X: [B, S, D]
   ↓
Linear: [B, S, D] → [B, S, D] (for Q, K, V)
   ↓
Split into heads: [B, H, S, d_k]
   ↓
Compute attention: for each head
   ↓
Concatenate heads: [B, S, D]
   ↓
Final linear: [B, S, D]
```

---

## 💡 总结

| 组件 | 作用 |
|------|------|
| `W_q`, `W_k`, `W_v` | 投影到 Query、Key、Value 空间 |
| 分头（Split） | 将向量分割为多个 head |
| attention score | 计算 token 之间的相关性 |
| 加权求和 | 使用 attention 权重聚合信息 |
| 合并头（Concat） | 将多个 head 的结果合并 |
| `W_o` | 最终整合输出 |

如下图所示 只关注最后两个维度 其他的 算是广播 类似于嵌套循环
---
      Q: [B, H, S, D]       ×       K^T: [B, H, D, S]
         └───────┬─────────┘            └───────┬─────────┘
                 ├─(matrix mul)                  ├─(dim swap)
                 ↓                               ↓
           result: [B, H, S, S]
## 📚 拓展学习建议

如果你感兴趣，还可以继续学习以下内容：
- 如何加入 mask（padding mask 或 causal mask）
- `torch.nn.MultiheadAttention` 的使用
- 自注意力 vs 交叉注意力的区别
- 使用 `einops` 更优雅地处理多维张量变换

---

如需我帮你继续扩展这份笔记，比如加入图解、动画演示、可视化 attention 分布等内容，请随时告诉我 😊