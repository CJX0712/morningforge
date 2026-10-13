# MorningForge 架构与设计说明

> 作者：晨星（CJX0712）  ·  版本：v1.0.0

## 1. 设计哲学

MorningForge 不是又一个「从零造轮子」的玩具库，而是**集大成**的产物：

1. **零重写**：每个模块先检索并复用世界级方案的设计（见下表），不重复发明已被证明正确的数学与工程。
2. **可验证**：每个算法至少有 1 条可被独立实现交叉验证的**不变量**。不变量是「真」的唯一仲裁——不靠肉眼，靠数值。
3. **零依赖**：纯 JavaScript ESM，浏览器与 Node 同码运行。无构建步骤、无 npm 安装、无 WASM。

## 2. 模块与「站在谁肩上」

| 模块 | 世界级方案 | 关键不变量（验证标尺） |
|------|-----------|----------------------|
| `core/matrix` | BLAS/LAPACK 矩阵运算语义 | `A·I=A`、`(AB)C=A(BC)` |
| `core/linalg.inverse` | Gauss-Jordan（部分主元）= LAPACK xGETRI | `A·A⁻¹=I`（残差 ≤ ε） |
| `core/linalg.svd` | 双边 Jacobi（Hestenes 1958）= LAPACK DGESVD 高精度路径 | `‖A−UΣVᵀ‖≤ε`、`UᵀU=I`、`VᵀV=I` |
| `core/linalg.eigSymmetric` | Jacobi 循环旋转 = LAPACK DSYEV | `‖Av−λv‖≤ε`、特征向量正交、实特征值 |
| `nn/autograd` | Karpathy micrograd + PyTorch autograd | `maxRelErr ≈ 1e-7`（中心差分 vs 解析梯度） |
| `nn/mlp` | Rumelhart 1986 MLP + Kaiming 初始化 + Adam（Kingma & Ba 2015） | 训练损失单调下降、收敛到低值 |
| `nn/transformer` | Vaswani 2017 + nanoGPT | 注意力权重行和=1、梯度检验通过 |
| `ml/pca` | Pearson / Jolliffe 2002 | 协方差对称 PSD、累计方差单调、重构误差随 k 递减 |
| `ml/kmeans` | Lloyd 1967 + k-means++（Arthur & Vassilvitskii 2007） | 惯性（SSE）单调非增 |

> **为何用纯净实现而非直接 import numpy/torch**：本库目标是「浏览器原生、零依赖、可被不变量逐条验证」的基元。工业库（需 Native/WASM）无法直接提供这一属性。生产中按上表替换为对应工业级实现即可——接口语义保持一致。

## 3. 数值方法要点

### 3.1 双边 Jacobi SVD
- 对 `A(m×n)` 迭代施加旋转 `A ← Jᵀ·A·J`，同时累积 `U ← U·J`、`V ← V·J`。
- 收敛判据：列间内积平方和 `≤ 1e-14`。
- 收敛后 `M` 列正交，**奇异值 = 列范数**，左奇异向量 = 归一化列；`V` 取左上 `n×n` 块（标准 SVD 中 `V` 为 `n×n`）。
- 精度达 `~1e-12`（优于经 `AᵀA` 的二次方法，后者会平方条件数）。

### 3.2 磁带式自动微分
- 标量计算图：每个 `Value` 记录子节点与反向闭包。
- `backward()` 拓扑排序后反向传播。
- 梯度检验：中心差分 `(f(x+ε)−f(x−ε))/(2ε)` 与解析梯度比较，`maxRelErr ≤ 1e-7` 为金标准。

### 3.3 k-means
- k-means++ 初始化保证 `Θ(log k)` 近似比。
- Lloyd 迭代：分配 → 更新 → 惯性（SSE）单调非增（不变量）。

## 4. 验证协议

`test/run.mjs` 为**不变量驱动**自检，零网络依赖，固定随机种子（`mulberry32(20261014)`），完全可复现。任一不变量失败即 `exit(1)`。CI（`.github/workflows/test.yml`）在每次 push 复跑。

## 5. 已知边界与升级路径

- 稠密实现，超大数据集请换工业库（scikit-learn / FAISS / torch）。
- SVD 采用 Jacobi（高精度但 O(n³) 常数较大）；超大规模用随机 SVD（Halko 2011）。
- 自动微分为标量图，批量张量场景换 PyTorch/JAX。
- GPU 推理不在本库范围；生产部署用 vLLM / SGLang / ONNX Runtime。
