// MorningForge · src/index.mjs
// 统一入口：聚合所有模块导出。
// 作者：晨星

export * from './core/matrix.mjs';
export * from './core/linalg.mjs';
export * from './core/random.mjs';
export * from './nn/autograd.mjs';
export * from './nn/mlp.mjs';
export * from './nn/transformer.mjs';
export * from './ml/pca.mjs';
export * from './ml/kmeans.mjs';

export const VERSION = '1.0.0';
export const AUTHOR = '晨星';
