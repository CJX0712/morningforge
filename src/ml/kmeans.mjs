// MorningForge · ml/kmeans.mjs
// k-means 聚类：k-means++ 初始化 + Lloyd 迭代。
// 设计蓝图：Lloyd 算法（MacQueen 1967）+ k-means++（Arthur & Vassilvitskii 2007）。
// 生产环境替换为 scikit-learn KMeans（Elkan 三角不等式加速）。
// 作者：晨星

// k-means++ 初始化
export function kmeansPP(X, k, rng = Math.random) {
  const n = X.length;
  const d = X[0].length;
  const centroids = [X[Math.floor(rng() * n)].slice()];
  while (centroids.length < k) {
    const dist2 = X.map((x) => {
      let best = Infinity;
      for (const c of centroids) {
        let s = 0;
        for (let j = 0; j < d; j++) {
          const diff = x[j] - c[j];
          s += diff * diff;
        }
        if (s < best) best = s;
      }
      return best;
    });
    const sum = dist2.reduce((a, b) => a + b, 0);
    let r = rng() * sum;
    let cum = 0;
    let pick = 0;
    for (let i = 0; i < n; i++) {
      cum += dist2[i];
      if (cum >= r) {
        pick = i;
        break;
      }
    }
    centroids.push(X[pick].slice());
  }
  return centroids;
}

// k-means 主过程。返回 { centroids, labels, inertia, inertiaHistory }
//   inertiaHistory 单调非增（Lloyd 不变量）。
export function kmeans(X, k, rng = Math.random, maxIter = 100) {
  const n = X.length;
  const d = X[0].length;
  let centroids = kmeansPP(X, k, rng);
  const labels = new Array(n).fill(0);

  const inertiaOf = (cents, labs) => {
    let s = 0;
    for (let i = 0; i < n; i++) {
      const c = cents[labs[i]];
      for (let j = 0; j < d; j++) {
        const diff = X[i][j] - c[j];
        s += diff * diff;
      }
    }
    return s;
  };

  const inertiaHistory = [];
  for (let iter = 0; iter < maxIter; iter++) {
    for (let i = 0; i < n; i++) {
      let best = Infinity;
      let bi = 0;
      for (let c = 0; c < k; c++) {
        let s = 0;
        for (let j = 0; j < d; j++) {
          const diff = X[i][j] - centroids[c][j];
          s += diff * diff;
        }
        if (s < best) {
          best = s;
          bi = c;
        }
      }
      labels[i] = bi;
    }
    const newC = centroids.map(() => new Array(d).fill(0));
    const cnt = new Array(k).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < d; j++) newC[labels[i]][j] += X[i][j];
      cnt[labels[i]]++;
    }
    let moved = false;
    for (let c = 0; c < k; c++) {
      if (cnt[c] === 0) continue;
      for (let j = 0; j < d; j++) {
        const nv = newC[c][j] / cnt[c];
        if (Math.abs(nv - centroids[c][j]) > 1e-12) moved = true;
        newC[c][j] = nv;
      }
    }
    centroids = newC;
    inertiaHistory.push(inertiaOf(centroids, labels));
    if (!moved) break;
  }
  return { centroids, labels, inertia: inertiaHistory[inertiaHistory.length - 1], inertiaHistory };
}
