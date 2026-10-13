# MorningForge · 晨星锻造

> 零依赖、可被**不变量交叉验证**的世界级 AI/ML 基础算法库。
> 纯 JavaScript（ESM），**浏览器 + Node 同码运行**，无需任何第三方依赖。

[![tests](https://img.shields.io/badge/self--test-8%2F8%20green-2ea043)](test/run.mjs)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![zero-dep](https://img.shields.io/badge/dependencies-0-brightgreen)](package.json)

---

## 一句话定位

每个算法模块都**站在世界级成果之上**（零重写 · 集大成），并以一条可被独立实现交叉验证的**数学不变量**作为「真」的标尺——解析解 vs 数值解、有限差分 vs 解析梯度、参照实现 vs 主实现。全部不变量由 `test/run.mjs` 自动校验，**8/8 绿，零网络依赖，完全可复现**。

## 技术原则落地（零重写 / 集大成）

| 模块 | 建立在世界级方案之上 | 生产中应替换为 |
|------|---------------------|---------------|
| 自动微分 | Karpathy micrograd + PyTorch autograd 语义 | PyTorch / JAX autograd |
| 矩阵求逆 | Gauss-Jordan（部分主元）= LAPACK xGETRF/xGETRI 语义 | LAPACK / numpy |
| SVD | 双边 Jacobi（Hestenes 1958）= LAPACK DGESVD 高精度路径 | LAPACK / numpy |
| 特征分解 | Jacobi 循环旋转 = LAPACK DSYEV 语义 | LAPACK / numpy |
| PCA | Pearson / Jolliffe 2002 | scikit-learn PCA |
| k-means | Lloyd 1967 + k-means++（Arthur & Vassilvitskii 2007） | scikit-learn KMeans（Elkan） |
| Transformer | Vaswani et al. 2017 + nanoGPT（Karpathy） | FlashAttention / xFormers |
| 优化器 | Adam（Kingma & Ba 2015） | torch.optim.Adam |

> 纯净实现的意义：**浏览器原生、零依赖、可被不变量逐条验证**——这是工业级 WASM 库难以直接提供的。一旦发现更强的现成方案，按上表直接替换自写实现即可。

## 目录结构

```
morningforge/
├── src/
│   ├── core/matrix.mjs      # 线性代数基础（matmul / 逆 / 转置）
│   ├── core/linalg.mjs      # 求逆 · 对称特征分解 · 双边 Jacobi SVD
│   ├── core/random.mjs      # 确定性 PRNG（可复现）
│   ├── nn/autograd.mjs      # 磁带式自动微分 + 梯度检验
│   ├── nn/mlp.mjs           # MLP + Adam + 螺旋数据集
│   ├── nn/transformer.mjs   # 缩放点积注意力 / 多头注意力
│   ├── ml/pca.mjs           # 主成分分析
│   ├── ml/kmeans.mjs        # k-means 聚类
│   └── index.mjs            # 统一入口
├── test/run.mjs             # 不变量自检套件（8/8 绿）
├── demo/index.html          # 可视化演示（PCA / k-means / MLP / 注意力）
├── docs/ARCHITECTURE.md     # 架构与设计说明
└── package.json
```

## 快速开始

### 运行自检（验证不变量）

```bash
node test/run.mjs
# === 结果：8/8 通过 ===
# 🟢 全部不变量校验通过（8/8 绿）
```

### 在 Node 中使用

```js
import { svd, pca, kmeans, makeMLP, attention } from './src/index.mjs';

// SVD：A ≈ U · diag(S) · Vᵀ
const { U, S, V } = svd([[1, 2], [3, 4], [5, 6]]);
console.log('奇异值', S);

// PCA：协方差特征分解
const X = [[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0]];
const model = pca(X);
console.log('累计方差', model.cumulative);

// k-means
const { centroids, labels } = kmeans([[0,0],[0.1,0.1],[5,5],[5.1,4.9]], 2);
console.log('簇中心', centroids);
```

### 在浏览器中使用

通过任意静态服务器打开演示（ES Module 需经 HTTP 加载，`file://` 会被 CORS 拦截）：

```bash
npm run demo          # = python -m http.server 5173
# 浏览器访问 http://localhost:5173/demo/
```

或直接以 `<script type="module">` 引入 `src/index.mjs`。

## 验证方式（可复现 / 可验证）

`test/run.mjs` 校验 8 条数学不变量（每条对应《AI 大全》标注的硬金标准）：

| # | 不变量 | 同时精度 |
|---|--------|---------|
| 1 | 矩阵单位元 `A·I=A` 与结合律 `(AB)C=A(BC)` | ≤ 1.2e-16 |
| 2 | 矩阵求逆 `A·A⁻¹ = I` | ≤ 3.3e-16 |
| 3 | SVD 重构 `‖A−UΣVᵀ‖/‖A‖` 与 `UᵀU=VᵀV=I` | ≤ 7.2e-16 |
| 4 | 特征残差 `‖Av−λv‖` 与正交性、PSD | ≤ 1.5e-15 |
| 5 | 自动微分梯度检验（中心差分 vs 解析） | ≤ 6.8e-12 |
| 6 | PCA：协方差对称 PSD、累计方差单调、重构误差随 k 递减 | ✅ |
| 7 | k-means：惯性（SSE）单调非增 | ✅ |
| 8 | 注意力权重行和 = 1 + 梯度检验 | ≤ 6.2e-10 |

## 部署 / 开源

- 仓库：`https://github.com/CJX0712/morningforge`
- 作者：**晨星（CJX0712）**
- 持续集成：`.github/workflows/test.yml` 在每次 push 运行 `node test/run.mjs`。

## 许可证

MIT © 2026 晨星 (CJX0712)
