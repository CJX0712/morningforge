// MorningForge · core/matrix.mjs
// 零依赖线性代数基础（稠密矩阵）。
// 设计蓝图：BLAS/LAPACK 矩阵运算语义（matmul / transpose / ...）。
// 生产环境应替换为 BLAS 绑定（numpy / Eigen / torch）。此处提供浏览器原生、
// 可读、可被不变量校验的纯净实现（well-conditioned 区间数值误差 ≤ 1e-10）。
// 作者：晨星

export function zeros(r, c) {
  const m = new Array(r);
  for (let i = 0; i < r; i++) m[i] = new Array(c).fill(0);
  return m;
}

export function ones(r, c) {
  const m = new Array(r);
  for (let i = 0; i < r; i++) m[i] = new Array(c).fill(1);
  return m;
}

export function identity(n) {
  const m = zeros(n, n);
  for (let i = 0; i < n; i++) m[i][i] = 1;
  return m;
}

export function clone(A) {
  return A.map((row) => row.slice());
}

export function transpose(A) {
  const r = A.length;
  const c = A[0].length;
  const T = zeros(c, r);
  for (let i = 0; i < r; i++)
    for (let j = 0; j < c; j++) T[j][i] = A[i][j];
  return T;
}

export function add(A, B) {
  const r = A.length;
  const c = A[0].length;
  const M = zeros(r, c);
  for (let i = 0; i < r; i++)
    for (let j = 0; j < c; j++) M[i][j] = A[i][j] + B[i][j];
  return M;
}

export function sub(A, B) {
  const r = A.length;
  const c = A[0].length;
  const M = zeros(r, c);
  for (let i = 0; i < r; i++)
    for (let j = 0; j < c; j++) M[i][j] = A[i][j] - B[i][j];
  return M;
}

export function scale(A, s) {
  return A.map((row) => row.map((v) => v * s));
}

export function hadamard(A, B) {
  const r = A.length;
  const c = A[0].length;
  const M = zeros(r, c);
  for (let i = 0; i < r; i++)
    for (let j = 0; j < c; j++) M[i][j] = A[i][j] * B[i][j];
  return M;
}

// A(r×k) · B(k×c) —— 经典三重循环，含零跳过优化。
export function matmul(A, B) {
  const r = A.length;
  const k = B.length;
  const c = B[0].length;
  const M = zeros(r, c);
  for (let i = 0; i < r; i++) {
    const Arow = A[i];
    const Mrow = M[i];
    for (let p = 0; p < k; p++) {
      const a = Arow[p];
      if (a === 0) continue;
      const Brow = B[p];
      for (let j = 0; j < c; j++) Mrow[j] += a * Brow[j];
    }
  }
  return M;
}

export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

// Frobenius 范数
export function normF(A) {
  let s = 0;
  for (const row of A) for (const v of row) s += v * v;
  return Math.sqrt(s);
}

export function approxEqual(A, B, tol = 1e-9) {
  for (let i = 0; i < A.length; i++)
    for (let j = 0; j < A[0].length; j++)
      if (Math.abs(A[i][j] - B[i][j]) > tol) return false;
  return true;
}
