// MorningForge · core/random.mjs
// 确定性随机数（mulberry32）。用于可复现实验与测试。
// 作者：晨星

// 设计蓝图：mulberry32 (George Marsaglia 系 PRNG)。生产环境替换为 numpy.random / PCG。
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
