// MorningForge · nn/autograd.mjs
// 磁带式自动微分（标量计算图）。
// 设计蓝图：Karpathy micrograd 的极简设计 + PyTorch autograd 语义。
// 生产环境替换为 PyTorch / JAX autograd（支持张量、反向模式、CUDA）。
// 此处提供零依赖、可被有限差分梯度校验的纯净实现（不变量：maxRelErr ≈ 1e-7）。
// 作者：晨星

export class Value {
  constructor(data, children = [], op = '') {
    this.data = data;
    this.grad = 0;
    this.children = children;
    this.op = op;
    this._backward = () => {};
  }

  add(other) {
    const o = other instanceof Value ? other : new Value(other);
    const out = new Value(this.data + o.data, [this, o], '+');
    out._backward = () => {
      this.grad += out.grad;
      o.grad += out.grad;
    };
    return out;
  }

  mul(other) {
    const o = other instanceof Value ? other : new Value(other);
    const out = new Value(this.data * o.data, [this, o], '*');
    out._backward = () => {
      this.grad += o.data * out.grad;
      o.grad += this.data * out.grad;
    };
    return out;
  }

  pow(p) {
    const out = new Value(Math.pow(this.data, p), [this], `**${p}`);
    out._backward = () => {
      this.grad += p * Math.pow(this.data, p - 1) * out.grad;
    };
    return out;
  }

  neg() {
    return this.mul(-1);
  }

  sub(other) {
    const o = other instanceof Value ? other : new Value(other);
    return this.add(o.neg());
  }

  div(other) {
    const o = other instanceof Value ? other : new Value(other);
    return this.mul(o.pow(-1));
  }

  relu() {
    const out = new Value(this.data > 0 ? this.data : 0, [this], 'relu');
    out._backward = () => {
      this.grad += (this.data > 0 ? 1 : 0) * out.grad;
    };
    return out;
  }

  sigmoid() {
    const s = 1 / (1 + Math.exp(-this.data));
    const out = new Value(s, [this], 'sigmoid');
    out._backward = () => {
      this.grad += s * (1 - s) * out.grad;
    };
    return out;
  }

  tanh() {
    const t = Math.tanh(this.data);
    const out = new Value(t, [this], 'tanh');
    out._backward = () => {
      this.grad += (1 - t * t) * out.grad;
    };
    return out;
  }

  exp() {
    const e = Math.exp(this.data);
    const out = new Value(e, [this], 'exp');
    out._backward = () => {
      this.grad += e * out.grad;
    };
    return out;
  }

  log() {
    const out = new Value(Math.log(this.data), [this], 'log');
    out._backward = () => {
      this.grad += (1 / this.data) * out.grad;
    };
    return out;
  }

  backward() {
    const topo = [];
    const visited = new Set();
    const build = (v) => {
      if (visited.has(v)) return;
      visited.add(v);
      for (const c of v.children) build(c);
      topo.push(v);
    };
    build(this);
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
  }
}

export function sum(values) {
  return values.reduce((a, b) => a.add(b), new Value(0));
}

export function mean(values) {
  return sum(values).div(values.length);
}

// 数值梯度校验：比较解析梯度与中心差分，返回最大相对误差。
// f(params) -> Value（标量损失），params 为叶子 Value[]。
// 对“解析与数值均≈0”的参数跳过（避免相对误差爆炸），阈值 lenient=1e-4 用于判定。
export function gradientCheck(f, params, eps = 1e-5) {
  for (const p of params) p.grad = 0;
  const loss = f(params);
  loss.backward();
  const analytic = params.map((p) => p.grad);

  const numeric = [];
  for (let i = 0; i < params.length; i++) {
    const orig = params[i].data;
    params[i].data = orig + eps;
    const lp = f(params).data;
    params[i].data = orig - eps;
    const lm = f(params).data;
    params[i].data = orig;
    numeric.push((lp - lm) / (2 * eps));
  }

  let maxRel = 0;
  for (let i = 0; i < params.length; i++) {
    const a = analytic[i];
    const n = numeric[i];
    if (Math.abs(a) < 1e-6 && Math.abs(n) < 1e-6) continue; // 双近零跳过
    const rel = Math.abs(a - n) / (Math.abs(a) + Math.abs(n) + 1e-12);
    if (rel > maxRel) maxRel = rel;
  }
  return maxRel;
}
