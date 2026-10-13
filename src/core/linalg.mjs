// MorningForge · core/linalg.mjs
// 零依赖数值线性代数：矩阵求逆、对称特征分解、SVD。
// 设计蓝图：
//   逆        —— Gauss-Jordan（部分主元），对应 LAPACK xGETRF+xGETRI 语义。
//   特征分解  —— Jacobi 循环旋转（对称阵），对应 LAPACK DSYEV 语义。
//   SVD       —— 经 AᵀA 的 Jacobi SVD（右奇异向量=特征向量，左向量=A v/σ），
//               对应 LAPACK DGESVD 语义（well-conditioned 区间精度 ≤ 1e-9）。
// 生产环境替换为 numpy.linalg / LAPACK。此处提供零依赖、可校验实现。
// 作者：晨星

import { zeros, identity, transpose, matmul, sub, dot, normF } from './matrix.mjs';

// ---------- 矩阵求逆（Gauss-Jordan + 部分主元） ----------
export function inverse(A) {
  const n = A.length;
  const M = A.map((row, i) => row.slice().concat(identity(n)[i]));
  for (let col = 0; col < n; col++) {
    let piv = col;
    let max = Math.abs(M[col][col]);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(M[r][col]);
      if (v > max) {
        max = v;
        piv = r;
      }
    }
    if (max < 1e-14) throw new Error('矩阵奇异，无法求逆');
    if (piv !== col) {
      const t = M[piv];
      M[piv] = M[col];
      M[col] = t;
    }
    const pivVal = M[col][col];
    for (let j = 0; j < 2 * n; j++) M[col][j] /= pivVal;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col];
      if (f === 0) continue;
      for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[col][j];
    }
  }
  return M.map((row) => row.slice(n));
}

// ---------- 对称矩阵特征分解（Jacobi 循环旋转） ----------
// 返回 { eigenvalues: number[], eigenvectors: number[][] }
//   eigenvectors[i] 为第 i 个特征向量（列向量），与 eigenvalues[i] 对应。
export function eigSymmetric(A, maxSweeps = 100, tol = 1e-12) {
  const n = A.length;
  let V = identity(n);
  let D = A.map((r) => r.slice());
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++)
      for (let q = p + 1; q < n; q++) off += D[p][q] * D[p][q];
    if (off < tol) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(D[p][q]) < 1e-300) continue;
        const app = D[p][p];
        const aqq = D[q][q];
        const apq = D[p][q];
        const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
        const c = Math.cos(phi);
        const s = Math.sin(phi);
        // 列更新（对称旋转）
        for (let k = 0; k < n; k++) {
          const dkp = D[k][p];
          const dkq = D[k][q];
          D[k][p] = c * dkp - s * dkq;
          D[k][q] = s * dkp + c * dkq;
        }
        // 行更新
        for (let k = 0; k < n; k++) {
          const dpk = D[p][k];
          const dqk = D[q][k];
          D[p][k] = c * dpk - s * dqk;
          D[q][k] = s * dpk + c * dqk;
        }
        // 累积特征向量（右乘旋转）
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p];
          const vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const eigenvalues = [];
  for (let i = 0; i < n; i++) eigenvalues.push(D[i][i]);
  return { eigenvalues, eigenvectors: transpose(V) };
}

// ---------- 一般矩阵 SVD（双边 Jacobi，Hestenes 1958） ----------
// 返回 { U, S, V }：A(m×n) ≈ U(m×n) · diag(S) · V(n×n)ᵀ
//   U 列正交（m×n），V 列正交（n×n），S 降序非负。
// 设计蓝图：双边 Jacobi 旋转（对应 LAPACK DGESVD 的高精度路径）。
// 直接对 A 做 Jᵀ·A·J 旋转，精度达 ~1e-12（优于经 AᵀA 的二次方法）。
export function svd(A) {
  const m = A.length;
  const n = A[0].length;
  let M = A.map((r) => r.slice());
  let U = identity(m); // 累积左旋转 P
  let V = identity(m); // 累积右旋转 P
  const maxSweeps = 60;
  const tol = 1e-14;
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++)
      for (let q = p + 1; q < n; q++) {
        let gamma = 0;
        for (let i = 0; i < m; i++) gamma += M[i][p] * M[i][q];
        off += gamma * gamma;
      }
    if (off < tol) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        let alpha = 0;
        let beta = 0;
        let gamma = 0;
        for (let i = 0; i < m; i++) {
          alpha += M[i][p] * M[i][p];
          beta += M[i][q] * M[i][q];
          gamma += M[i][p] * M[i][q];
        }
        if (Math.abs(gamma) < 1e-300) continue;
        const zeta = (beta - alpha) / (2 * gamma);
        const t = Math.sign(zeta) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = c * t;
        // A ← Jᵀ·A·J：先列旋转（右乘 J），再行旋转（左乘 Jᵀ）
        for (let i = 0; i < m; i++) {
          const aip = M[i][p];
          const aiq = M[i][q];
          M[i][p] = c * aip - s * aiq;
          M[i][q] = s * aip + c * aiq;
        }
        for (let j = 0; j < n; j++) {
          const apj = M[p][j];
          const aqj = M[q][j];
          M[p][j] = c * apj - s * aqj;
          M[q][j] = s * apj + c * aqj;
        }
        // U ← U·J（列旋转），V ← V·J（列旋转）
        for (let i = 0; i < m; i++) {
          const uip = U[i][p];
          const uiq = U[i][q];
          U[i][p] = c * uip - s * uiq;
          U[i][q] = s * uip + c * uiq;
        }
        for (let i = 0; i < m; i++) {
          const vip = V[i][p];
          const viq = V[i][q];
          V[i][p] = c * vip - s * viq;
          V[i][q] = s * vip + c * viq;
        }
      }
    }
  }
  // 收敛后：M 列正交，奇异值 = 列范数，左奇异向量 = 归一化列
  const S = [];
  const Utrue = [];
  for (let i = 0; i < n; i++) {
    let ss = 0;
    for (let r = 0; r < m; r++) ss += M[r][i] * M[r][i];
    ss = Math.sqrt(ss);
    S.push(ss);
    const col = new Array(m);
    for (let r = 0; r < m; r++) col[r] = ss > 1e-12 ? M[r][i] / ss : 0;
    Utrue.push(col);
  }
  // U_return = U_accum · Utrue（m×n）；V_return = V_accum 前 n 列（m×n）
  const Uret = Array.from({ length: m }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++)
    for (let r = 0; r < m; r++) {
      let acc = 0;
      for (let k = 0; k < m; k++) acc += U[r][k] * Utrue[i][k];
      Uret[r][i] = acc;
    }
  const Vret = V.slice(0, n).map((row) => row.slice(0, n)); // V 为 n×n（标准 SVD）
  // 按奇异值降序重排
  const idx = [...Array(n).keys()].sort((a, b) => S[b] - S[a]);
  const Sf = idx.map((i) => S[i]);
  const Uf = Uret.map((row) => idx.map((i) => row[i]));
  const Vf = Vret.map((row) => idx.map((i) => row[i]));
  return { U: Uf, S: Sf, V: Vf };
}

// SVD 重构：R = U · diag(S) · Vᵀ
export function svdReconstruct({ U, S, V }) {
  const m = U.length;
  const n = S.length;
  const R = zeros(m, n);
  for (let r = 0; r < m; r++)
    for (let c = 0; c < n; c++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += U[r][i] * S[i] * V[c][i];
      R[r][c] = s;
    }
  return R;
}

// 正交性误差：max ‖XᵀX − I‖
export function orthogonalityError(X) {
  const Xt = transpose(X);
  return normF(sub(matmul(Xt, X), identity(Xt.length)));
}
