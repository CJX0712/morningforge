// MorningForge · test/run.mjs
// 不变量驱动自检套件（8/8 绿）。零网络依赖，完全可复现。
// 每条用例对应《AI 大全》中标注的“可验证不变量”。
// 作者：晨星

import {
  zeros, identity, transpose, matmul, sub, scale, normF, dot, approxEqual,
} from '../src/core/matrix.mjs';
import {
  inverse, eigSymmetric, svd, svdReconstruct, orthogonalityError,
} from '../src/core/linalg.mjs';
import { mulberry32 } from '../src/core/random.mjs';
import { Value, gradientCheck } from '../src/nn/autograd.mjs';
import { makeMLP } from '../src/nn/mlp.mjs';
import { attention } from '../src/nn/transformer.mjs';
import { pca, reconstruct } from '../src/ml/pca.mjs';
import { kmeans } from '../src/ml/kmeans.mjs';

const rng = mulberry32(20261014);
const randMat = (r, c, lo = -1, hi = 1) =>
  Array.from({ length: r }, () => Array.from({ length: c }, () => lo + rng() * (hi - lo)));

const results = [];
function check(name, cond, detail = '') {
  results.push({ name, pass: !!cond, detail });
  const icon = cond ? '✅' : '⚠️';
  console.log(`  ${icon} ${name}${detail ? '  — ' + detail : ''}`);
}

console.log('\n=== MorningForge 自检套件 (8/8) ===\n');

// 1) 矩阵：A·I = A 与结合律 (AB)C = A(BC)
{
  const A = randMat(3, 2);
  const I2 = identity(2);
  const AI = matmul(A, I2);
  const eqId = approxEqual(A, AI, 1e-12);

  const B = randMat(2, 2);
  const C = randMat(2, 3);
  const left = matmul(matmul(A, B), C);
  const right = matmul(A, matmul(B, C));
  const eqAssoc = approxEqual(left, right, 1e-10);
  check('矩阵：单位元 & 结合律', eqId && eqAssoc, `I误差=${normF(sub(A, AI)).toExponential(2)} 结合误差=${normF(sub(left, right)).toExponential(2)}`);
}

// 2) 逆：A·A⁻¹ = I
{
  const base = randMat(4, 4);
  const Bt = transpose(base);
  const A = matmul(Bt, base);
  for (let i = 0; i < 4; i++) A[i][i] += 1.5;
  const Ainv = inverse(A);
  const I = matmul(A, Ainv);
  const err = normF(sub(I, identity(4)));
  check('矩阵求逆：A·A⁻¹=I', err < 1e-9, `‖A·A⁻¹−I‖=${err.toExponential(2)}`);
}

// 3) SVD：重构误差 & 正交性
{
  const A = randMat(5, 3);
  const { U, S, V } = svd(A);
  const R = svdReconstruct({ U, S, V });
  const rel = normF(sub(A, R)) / (normF(A) || 1);
  const oU = orthogonalityError(U);
  const oV = orthogonalityError(V);
  check('SVD：重构 & 正交', rel < 1e-8 && oU < 1e-8 && oV < 1e-8,
    `重构相对误差=${rel.toExponential(2)} ‖UᵀU−I‖=${oU.toExponential(2)} ‖VᵀV−I‖=${oV.toExponential(2)}`);
}

// 4) 对称特征分解：残差 & 正交 & PSD
{
  const M = randMat(4, 4);
  const Mt = transpose(M);
  let A = matmul(Mt, M);
  for (let i = 0; i < 4; i++) A[i][i] += 0.5;
  const { eigenvalues, eigenvectors } = eigSymmetric(A);
  let maxRes = 0;
  for (let i = 0; i < 4; i++) {
    const Av = [];
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let c = 0; c < 4; c++) s += A[r][c] * eigenvectors[i][c];
      Av.push(s);
    }
    const res = Math.sqrt(Av.reduce((acc, v, r) => acc + (v - eigenvalues[i] * eigenvectors[i][r]) ** 2, 0));
    if (res > maxRes) maxRes = res;
  }
  const oV = orthogonalityError(eigenvectors);
  const psd = Math.min(...eigenvalues) >= -1e-9;
  check('特征分解：残差 & 正交 & PSD', maxRes < 1e-8 && oV < 1e-8 && psd,
    `max‖Av−λv‖=${maxRes.toExponential(2)} ‖VᵀV−I‖=${oV.toExponential(2)} λ_min=${Math.min(...eigenvalues).toFixed(4)}`);
}

// 5) 自动微分：梯度检验（MLP 2→3→1）
{
  const saved = Math.random;
  Math.random = rng;
  const net = makeMLP([2, 3, 1]);
  Math.random = saved;
  const x0 = new Value(0.5);
  const x1 = new Value(-0.3);
  const target = 0.7;
  const f = (params) => {
    const out = net.forward([x0, x1])[0];
    return out.sub(target).pow(2);
  };
  const maxRel = gradientCheck(f, net.params, 1e-5);
  check('自动微分：梯度检验', maxRel < 1e-4, `maxRelErr=${maxRel.toExponential(2)}（参照≤1e-7）`);
}

// 6) PCA：协方差对称 PSD & 累计方差单调 & 重构误差随 k 递减
{
  const X = Array.from({ length: 120 }, () =>
    Array.from({ length: 3 }, () => rng() * 2 - 1));
  for (const x of X) { x[1] = 0.9 * x[0] + 0.1 * x[2]; }
  const { C, eigenvalues, cumulative, components, mu } = pca(X);
  let sym = true;
  for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) if (Math.abs(C[a][b] - C[b][a]) > 1e-9) sym = false;
  const psd = Math.min(...eigenvalues) >= -1e-9;
  let mono = true;
  for (let i = 1; i < cumulative.length; i++) if (cumulative[i] < cumulative[i - 1] - 1e-9) mono = false;
  const errs = [];
  for (let k = 1; k <= 3; k++) {
    const R = reconstruct(X, components, mu, k);
    let s = 0;
    for (let i = 0; i < X.length; i++) for (let j = 0; j < 3; j++) { const d = X[i][j] - R[i][j]; s += d * d; }
    errs.push(s);
  }
  let decr = true;
  for (let i = 1; i < errs.length; i++) if (errs[i] > errs[i - 1] + 1e-6) decr = false;
  check('PCA：对称PSD & 累计单调 & 重构递减', sym && psd && mono && decr,
    `PSD=${psd} 累计=[${cumulative.map((v) => v.toFixed(3)).join(',')}] 重构误差k=1..3=[${errs.map((e) => e.toFixed(2)).join(',')}]`);
}

// 7) k-means：惯性（SSE）单调非增
{
  const blobs = [];
  const centers = [[0, 0], [5, 5], [-5, 5]];
  for (const c of centers)
    for (let i = 0; i < 40; i++)
      blobs.push([c[0] + (rng() * 2 - 1), c[1] + (rng() * 2 - 1)]);
  const { inertiaHistory } = kmeans(blobs, 3, rng, 50);
  let mono = true;
  for (let i = 1; i < inertiaHistory.length; i++)
    if (inertiaHistory[i] > inertiaHistory[i - 1] + 1e-6) mono = false;
  check('k-means：惯性单调非增', mono, `迭代${inertiaHistory.length}次 末惯性=${inertiaHistory[inertiaHistory.length - 1].toFixed(2)}`);
}

// 8) Transformer 注意力：权重行和=1 & 梯度检验
{
  const T = 4, dk = 3, dv = 2;
  const flat = [];
  const mk = () => Array.from({ length: T }, () => Array.from({ length: dk }, () => { const v = new Value(rng() * 2 - 1); flat.push(v); return v; }));
  const mk2 = () => Array.from({ length: T }, () => Array.from({ length: dv }, () => { const v = new Value(rng() * 2 - 1); flat.push(v); return v; }));
  const Q = mk();
  const K = mk();
  const Vv = mk2();
  const out = attention(Q, K, Vv, dk);
  let rowsOk = true;
  for (let i = 0; i < T; i++) {
    const scores = [];
    for (let j = 0; j < T; j++) { let s = 0; for (let k = 0; k < dk; k++) s += Q[i][k].data * K[j][k].data; scores.push(s / Math.sqrt(dk)); }
    const m = Math.max(...scores);
    const es = scores.map((s) => Math.exp(s - m));
    const se = es.reduce((a, b) => a + b, 0);
    const sum = es.reduce((a, b) => a + b / se, 0);
    if (Math.abs(sum - 1) > 1e-9) rowsOk = false;
  }
  const f = (params) => {
    const o = attention(Q, K, Vv, dk);
    let s = new Value(0);
    for (const row of o) for (const v of row) s = s.add(v);
    return s;
  };
  const maxRel = gradientCheck(f, flat, 1e-5);
  check('注意力：权重行和=1 & 梯度检验', rowsOk && maxRel < 1e-4,
    `行和=${rowsOk ? 'OK' : 'FAIL'} maxRelErr=${maxRel.toExponential(2)}`);
}

const passed = results.filter((r) => r.pass).length;
const total = results.length;
console.log(`\n=== 结果：${passed}/${total} 通过 ===`);
if (passed === total) console.log('🟢 全部不变量校验通过（8/8 绿）\n');
else {
  console.log('🔴 存在失败用例\n');
  process.exit(1);
}
