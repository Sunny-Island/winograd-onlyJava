## 配置环境

	a. 需要包含测试平台的硬件、软件环境信息，保证可复现性；

## 第一题报告
###　特性分析
- 缓存命中缓存一致性
### 优化思路
— 并行
— SIMD
算法优化
缓存/cache miss
# 实现/工程细节
缓存
矩阵simd avx128-256
多线程 无效到有效 分配
初始化速度
perf1 调优，寻找瓶颈

引入mkl

升级到avx512 win4x3优化 && 3d矩阵乘法优化

perf2 寻找瓶颈

## 参考资料

## 第二题报告
如何测得正确的带宽
async 分块 压缩 多进程

## 总结


