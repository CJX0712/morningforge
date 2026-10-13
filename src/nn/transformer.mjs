// MorningForge · nn/transformer.mjs
// 缩放点积注意力（单头 / 多头），标量图实现，便于梯度校验与可视化。
// 设计蓝图：Attention Is All You Need (Vaswani et al. 2017) + nanoGPT (Karpathy)。
// 生产环境替换为 FlashAttention / xFormers CUDA kernel。
// 作者：晨星

import { Value, sum } from './autograd.mjs';

// 单头缩放点积注意力。
//   Q,K: T×dk 的 Value 数组；V: T×dv 的 Value 数组。
// 返回 out: T×dv。
export function attention(Q, K, V, dk) {
  const T = Q.length;
  const dv = V[0].length;
  const out = [];
  for (let i = 0; i < T; i++) {
    const scores = [];
    for (let j = 0; j < T; j++) {
      let s = new Value(0);
      for (let k = 0; k < dk; k++) s = s.add(Q[i][k].mul(K[j][k]));
      s = s.div(Math.sqrt(dk));
      scores.push(s);
    }
    const m = Math.max(...scores.map((x) => x.data));
    const mval = new Value(m);
    const es = scores.map((s) => s.sub(mval).exp());
    const sumE = es.reduce((a, b) => a.add(b), new Value(0));
    const weights = es.map((e) => e.div(sumE)); // 每行和为 1（softmax 不变量）
    const o = [];
    for (let k = 0; k < dv; k++) {
      let acc = new Value(0);
      for (let j = 0; j < T; j++) acc = acc.add(weights[j].mul(V[j][k]));
      o.push(acc);
    }
    out.push(o);
  }
  return out;
}

// 多头注意力：heads 个单头拼接后线性投影。
//   Wout: dv_out× (heads*dv) 的 Value 矩阵（投影权重）。
export function multiHeadAttention(Q, K, V, dk, heads, Wout, dvOut) {
  const T = Q.length;
  const headOuts = [];
  for (let h = 0; h < heads; h++) {
    // 第 h 个头使用 Q/K/V 的 [h*dk, (h+1)*dk) 子维度
    const Qh = Q.map((row) => row.slice(h * dk, (h + 1) * dk));
    const Kh = K.map((row) => row.slice(h * dk, (h + 1) * dk));
    const Vh = V.map((row) => row.slice(h * dk, (h + 1) * dk));
    headOuts.push(attention(Qh, Kh, Vh, dk));
  }
  // 拼接：concat[h][i][k] = headOuts[h][i][k]
  const out = [];
  for (let i = 0; i < T; i++) {
    const concat = [];
    for (let h = 0; h < heads; h++)
      for (let k = 0; k < dvOut / heads; k++) concat.push(headOuts[h][i][k]);
    const projected = [];
    for (let o = 0; o < dvOut; o++) {
      let s = new Value(0);
      for (let j = 0; j < concat.length; j++) s = s.add(Wout[o][j].mul(concat[j]));
      projected.push(s);
    }
    out.push(projected);
  }
  return out;
}

// 注意力权重（仅数值，用于可视化 / 不变量校验）。
//   返回 T×T：weights[i][j] = softmax_i 对第 j 个 token 的注意力。
export function attentionWeights(Q, K, dk) {
  const T = Q.length;
  const W = [];
  for (let i = 0; i < T; i++) {
    const scores = [];
    for (let j = 0; j < T; j++) {
      let s = 0;
      for (let k = 0; k < dk; k++) s += Q[i][k] * K[j][k];
      scores.push(s / Math.sqrt(dk));
    }
    const m = Math.max(...scores);
    const es = scores.map((s) => Math.exp(s - m));
    const se = es.reduce((a, b) => a + b, 0);
    W.push(es.map((e) => e / se));
  }
  return W;
}
