// MorningForge · nn/mlp.mjs
// 多层感知机 + Adam 优化器 + 螺旋数据集（端到端可训练、可验证）。
// 设计蓝图：
//   网络      —— 标准 MLP（Rumelhart et al. 1986）+ Kaiming 初始化（He et al. 2015）。
//   优化器    —— Adam（Kingma & Ba 2015）。
//   损失      —— 稳定 log-sum-exp 交叉熵。
// 生产环境替换为 PyTorch nn.Module / torch.optim.Adam。
// 作者：晨星

import { Value } from './autograd.mjs';

// 构造 MLP；sizes 如 [2,16,16,3]。返回 { params, forward, W, b }。
export function makeMLP(sizes) {
  const params = [];
  const W = [];
  const b = [];
  for (let l = 0; l < sizes.length - 1; l++) {
    const w = [];
    const gain = Math.sqrt(2 / sizes[l]); // Kaiming uniform
    for (let i = 0; i < sizes[l + 1]; i++) {
      const row = [];
      for (let j = 0; j < sizes[l]; j++) {
        const v = new Value(gain * (Math.random() * 2 - 1));
        row.push(v);
        params.push(v);
      }
      w.push(row);
    }
    const bb = [];
    for (let i = 0; i < sizes[l + 1]; i++) {
      const v = new Value(0);
      bb.push(v);
      params.push(v);
    }
    W.push(w);
    b.push(bb);
  }

  function forward(x) {
    let h = x.map((v) => (v instanceof Value ? v : new Value(v)));
    for (let l = 0; l < W.length; l++) {
      const out = [];
      const last = l === W.length - 1;
      for (let i = 0; i < W[l].length; i++) {
        let s = b[l][i];
        for (let j = 0; j < W[l][i].length; j++) s = s.add(W[l][i][j].mul(h[j]));
        out.push(last ? s : s.relu());
      }
      h = out;
    }
    return h;
  }

  return { params, forward, W, b };
}

// Adam 一步更新（就地修改 params.data）。
export function adamStep(params, lr = 0.05, b1 = 0.9, b2 = 0.999, eps = 1e-8, t = 1, m = null, v = null) {
  if (!m) m = params.map(() => 0);
  if (!v) v = params.map(() => 0);
  const bc1 = 1 - Math.pow(b1, t);
  const bc2 = 1 - Math.pow(b2, t);
  for (let i = 0; i < params.length; i++) {
    const g = params[i].grad;
    m[i] = b1 * m[i] + (1 - b1) * g;
    v[i] = b2 * v[i] + (1 - b2) * g * g;
    const mh = m[i] / bc1;
    const vh = v[i] / bc2;
    params[i].data -= (lr * mh) / (Math.sqrt(vh) + eps);
  }
  return { m, v };
}

// 螺旋数据集（经典分类基准）。X: [x1,x2]，y: 类别。
export function makeSpiral(n = 100, classes = 3, rng = Math.random) {
  const X = [];
  const y = [];
  for (let c = 0; c < classes; c++) {
    for (let i = 0; i < n; i++) {
      const t = (i / n) * 2.5 + c * ((2 * Math.PI) / classes);
      const x1 = t * Math.cos(t) + 0.18 * (rng() * 2 - 1);
      const x2 = t * Math.sin(t) + 0.18 * (rng() * 2 - 1);
      X.push([x1, x2]);
      y.push(c);
    }
  }
  return { X, y };
}

// 训练 MLP（全批量）。返回 { net, losses, accuracy }。
export function trainMLP(sizes = [2, 16, 16, 3], iters = 400, lr = 0.1, rng = Math.random) {
  const net = makeMLP(sizes);
  const { X, y } = makeSpiral(100, sizes[sizes.length - 1], rng);
  let m = null;
  let v = null;
  const losses = [];
  let lastAcc = 0;
  for (let it = 0; it < iters; it++) {
    let loss = new Value(0);
    let correct = 0;
    for (let s = 0; s < X.length; s++) {
      const logits = net.forward(X[s].map((val) => new Value(val)));
      const maxL = Math.max(...logits.map((l) => l.data));
      const mval = new Value(maxL);
      const es = logits.map((l) => l.sub(mval).exp());
      const sumE = es.reduce((a, b) => a.add(b), new Value(0));
      const logsumexp = sumE.log().add(mval);
      const p = es[y[s]].div(sumE);
      loss = loss.add(p.log().mul(-1));
      let pred = 0;
      for (let idx = 1; idx < logits.length; idx++)
        if (logits[idx].data > logits[pred].data) pred = idx;
      if (pred === y[s]) correct++;
    }
    loss = loss.div(X.length);
    for (const p of net.params) p.grad = 0;
    loss.backward();
    const step = adamStep(net.params, lr, 0.9, 0.999, 1e-8, it + 1, m, v);
    m = step.m;
    v = step.v;
    losses.push(loss.data);
    lastAcc = correct / X.length;
    if (it % 50 === 0)
      console.log(`    iter ${it}: loss=${loss.data.toFixed(4)} acc=${(lastAcc * 100).toFixed(1)}%`);
  }
  return { net, losses, accuracy: lastAcc };
}
