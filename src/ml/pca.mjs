// MorningForge · ml/pca.mjs
// 主成分分析（基于协方差矩阵特征分解）。
// 设计蓝图：Pearson PCA（Jolliffe 2002）。
// 生产环境替换为 scikit-learn PCA（随机 SVD / ARPACK 加速）。
// 作者：晨星

import { transpose } from '../core/matrix.mjs';
import { eigSymmetric } from '../core/linalg.mjs';

// X: n×d。返回 { mu, C, eigenvalues, explained, cumulative, components }
//   components[c]：第 c 主成分方向（单位向量，按特征值降序）。
export function pca(X) {
  const n = X.length;
  const d = X[0].length;
  const mu = new Array(d).fill(0);
  for (const x of X) for (let j = 0; j < d; j++) mu[j] += x[j];
  for (let j = 0; j < d; j++) mu[j] /= n;

  const Xc = X.map((x) => x.map((v, j) => v - mu[j]));

  // 协方差 d×d
  const C = [];
  for (let a = 0; a < d; a++) {
    C.push(new Array(d).fill(0));
    for (let b = 0; b < d; b++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += Xc[i][a] * Xc[i][b];
      C[a][b] = s / (n - 1);
    }
  }

  const { eigenvalues, eigenvectors } = eigSymmetric(C);
  const idx = [...Array(d).keys()].sort((i, j) => eigenvalues[j] - eigenvalues[i]);
  const totalVar = eigenvalues.reduce((acc, e) => acc + Math.max(e, 0), 0);

  const explained = idx.map((i) => Math.max(eigenvalues[i], 0) / (totalVar || 1));
  const cumulative = [];
  let acc = 0;
  for (const i of idx) {
    acc += Math.max(eigenvalues[i], 0) / (totalVar || 1);
    cumulative.push(acc);
  }
  const components = idx.map((i) => eigenvectors[i]);
  return { mu, C, eigenvalues: idx.map((i) => eigenvalues[i]), explained, cumulative, components };
}

// 用前 k 个主成分重构（含均值）。
export function reconstruct(X, components, mu, k) {
  const n = X.length;
  const d = X[0].length;
  const R = X.map(() => mu.slice());
  for (let i = 0; i < n; i++) {
    const scores = [];
    for (let c = 0; c < k; c++) {
      let s = 0;
      for (let j = 0; j < d; j++) s += (X[i][j] - mu[j]) * components[c][j];
      scores.push(s);
    }
    for (let c = 0; c < k; c++)
      for (let j = 0; j < d; j++) R[i][j] += scores[c] * components[c][j];
  }
  return R;
}
