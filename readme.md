<!-- TOC -->

- [2021 九坤并行程序优化大赛初赛报告-OnlyJava 组](#2021-九坤并行程序优化大赛初赛报告-onlyjava-组)
    - [组员](#组员)
    - [配置环境](#配置环境)
        - [硬件环境](#硬件环境)
        - [软件环境](#软件环境)
    - [Winograd](#winograd)
        - [应用特性分析](#应用特性分析)
        - [优化思路](#优化思路)
        - [优化实现及工程细节](#优化实现及工程细节)
        - [测试结果](#测试结果)
        - [参考资料](#参考资料)
    - [H5bench](#h5bench)
        - [应用特性分析](#应用特性分析-1)
            - [影响 IO 的因素](#影响-io-的因素)
            - [如何测得正确的带宽](#如何测得正确的带宽)
        - [优化思路](#优化思路-1)
        - [优化实现及工程细节](#优化实现及工程细节-1)
        - [测试结果](#测试结果-1)
        - [参考资料](#参考资料-1)
    - [总结](#总结)

<!-- /TOC -->

# 2021 九坤并行程序优化大赛初赛报告-OnlyJava 组

## 组员
* 赵家贝 清华大学软件学院研二 zjb20@mails.tsinghua.edu.cn
* 谭新宇 清华大学软件学院研二 tanxinyu@apache.org
* 宋一唱 清华大学软件学院研二

## 配置环境

### 硬件环境
* CPU：Intel(R) Xeon(R) Gold 6348 CPU @ 2.60GHz
* 内存：768 GB
* 磁盘：需参照宿主机具体磁盘配置，文件系统为 Overlay

### 软件环境
* 操作系统：CentOS Linux release 8.4.2105
* GCC：8.4.1
* MKL：oneAPI mkl-2021.4.0 
* MPI：openmpi-4.1.1

## Winograd

### 应用特性分析

Winograd 算法来自于 CVPR 2016 的一篇 paper：[Fast Algorithms for Convolutional Neural Networks](https://arxiv.org/abs/1509.09308)。Winograd 算法的主要思路便是通过减少乘法次数增多加法次数来实现提速。总体上来看这是一个计算密集的场景，因而优化便在于如果最大程度压榨出 CPU 的性能。

### 优化思路

* 并行

卷积运算的过程是可以进行并行计算的，比如示例代码中给出的计算中，每一个 channel 的计算就是完全互不干扰，可以并行的。要想加速卷积运算，需要充分利用计算平台多核的优势，尽可能在每一个环节把计算并行化。

我们在算法中通过 openmp 库，进行了并行计算的尝试。

* SIMD 向量化

单指令流多数据流（SIMD）是能够复制多个操作数，并把它们打包在大型寄存器的一组指令集。向量化计算是一种常用于加速计算密集的并行任务的优化方法。

1997 年，x86 扩展出了 MMX 指令集，伴随着 8 个向量寄存器，首开了向量化计算的先河。之后 x86 又扩展出了 SSE 和 AVX 指令集。SSE 和 AVX 各有 16 个寄存器，同一个寄存器可以装载多个数据单元，并且可以用相应的指令在同一个时钟周期内进行并行运算，随着指令集的升级，每个时钟周期可以并行的单元数也在不断增加。

在我们的服务器和赛方提供的服务器上，我们依次实验了 AVX128，AVX256 和 AVX512 指令集。

- 充分利用缓存

在计算密集的场景中，CPU 的访存时延往往会成为瓶颈。因此要尽可能利用好 L2/L3 缓存。落地到代码中便是精心设计数据结构在内存中的布局，尽量进行批量化的计算等等。此外还需要避免多线程环境下的伪共享问题，以避免程序性能被大幅降低。

* 编译优化

编译优化是计算密集场景中非常重要的优化工作。尽管我们可以按照底层的硬件属性写出尽可能高效的代码，但一个成熟的编译器往往考虑的比我们更多。这些优化包括但不限于循环展开，并行优化，向量化，寄存器变量，矩阵分块等等。

因此，我们对目前两种最流行的 C 编译器 GCC 和 LLVM 都进行了性能验证。

* 算法优化

在示例代码中的 Winograd 算法实际上是一种一维算法，需要逐个对每个图片每个 channel 的每一行每一列与每个卷积核相乘，因此有大量的 for 循环。通过查阅文献，我们发现在 3D 卷积运算中，相对于示例代码我们可以做出三个优化：
  - 转换 1D 的 winograd 算法到 2D。比如示例的 winograd(2,3) 可以转换成 winograd(2x2, 3x3), 可以同时计算得到一个 (2x2) 的输出。
  - 转换 1D 的 winograd 算法到 3D。相关文献指出，在 3D 的矩阵运算中，可以把 winograd 算法分成 4 步，分别是 Input transform、Filter transform、Matrix Multiplication、Output transform。
![](https://s2.ax1x.com/2019/04/29/E1qSoR.png)
  - 扩展 winograd(2,3) 算法到 winograd(4,3)。根据相关文献指出，4x3 算法能更多地减少乘法运算的比例，从而带来更好的加速效果。而且 4x3 将矩阵分块更大一些，对增加缓存命中也相对友好。

根据文献的实验结果，3D-winograd(2,3) 算法是最节能的，适合在 FPGA 等边缘设备上部署。但 3D-winograd(4,3) 算法前向传播是最快的，因此我们最终采取了 3D-winograd(4,3) 算法。

### 优化实现及工程细节

* 缓存

在优化时，我们按照程序中的提示，先从优化矩阵计算出发。示例代码的矩阵运算显然是一种对缓存的低效利用，每次读取一段内存，但只用到了其中的第一个数据，下次循环又要去读取新的内存。通过交换循环的顺序，即可提升对缓存的利用效率。此优化有近 14 倍的提升 **(0.22GFlops->3.05GFlops)**

* SIMD AVX128

把矩阵运算划分成一部分 4x4 矩阵运算，不能被 4 整除的部分单独做 1x1 的矩阵运算。并通过 AVX128 指令集计算 4x4 矩阵运算，在 AVX128 中一个寄存器可以存储 4 个单精度浮点数，即可以同时对它们做向量化计算。在 4x4 的运算中没有循环，全部做循环展开做计算。同时利用 FMA 指令引入积和熔加运算，进一步压缩指令。此优化有近 4 倍的提升。 **(3.05GFlops->12.88 GFlops)**

* 并行
  
此时，我们尝试通过引入 openmp 多线程进行矩阵运算，但并行的性能一直不如串行，甚至经过实验发现有线程数加倍性能减半的现象。经过分析发现在示例代码中有大量的小矩阵运算，比如在第一部分逐 channel 逐卷积核的矩阵相乘中有大量的大小为 4 的矩阵，矩阵太小导致分给每个线程的任务过少，线程本身的调度开销大于多线程带来的收益，因此多线程在这种运算中是一种负优化。

为了引入并行加快速度，我们单独为中间的大矩阵运算编写了加入了多线程的函数，小矩阵运算则不加多线程。此优化有近 5 倍的提升。**(12.88GFlops->57.11GFlops)**

* Perf 调优

接着，我们利用 Perf profile 了程序各部分的耗时分布。发现如下的问题：
1. 矩阵初始化置 0 的耗时占比不低，此处可以优化为 memset 的批量置 0。此优化有 6% 的提升 **(57.11GFlops->60.86GFlops)**
2. winograd 函数中大矩阵计算前后的两个部分耗时过长，这一部分属于算法设计问题，应该对整个算法进行调整。下文会进行介绍。
  
* SIMD AVX256

此时我们将指令集升级为了 AVX256。在 AVX256 中每个寄存器可以存储 8 个单精度浮点数，因此将原来 4x4 的矩阵运算更换为了 8x8 的矩阵运算。此优化有 1.6 倍的提升。**(60.86GFlops->97GFlops)**

* 编译器

以上的工作均是在 gcc8 版本上进行 O3 优化测得的，而 gcc 截止今日的最新版本已升级为 11.2.0，可以预见的是，新版本的 gcc 的调优性能会更突出，为此我们升级到了最新的 gcc 11.2.0 版本并继续用 O3 优化进行了测试。此工作提升了近 10% 的性能。**(97-106GFlops)**

此外，我们也尝试了最新版本的 Clang/LLVM 编译器，使用 O3 版本编译后发现性能甚至不如 gcc8。虽然我们的测试不够充分，但我们猜测 LLVM 为了兼容性相比 gcc 牺牲调了一定的优化空间。就如果 Java 之于 C/C++ 一样。

接着我们将以上优化搬迁到了赛方提供的服务器上。由于硬件性能的提升，测试结果提升到了 **170GFlops**。

* SIMD AVX256 && win4x3 算法 && 引入 MKL

论文中的算法描述如下所示：

![](https://s2.ax1x.com/2019/05/22/VpzRn1.png)

最后在赛方提供的服务器上，我们参照论文和相关资料重写了 winograd 算法，更新成 3D-winograd(4x3) 算法。以下我们分四个部分详细介绍该算法的实现：

1. Input transform

首先把输入矩阵拆分成 tiles，在这里 tiles 是长宽均为 6 的矩阵，在拆分过程中为了充分发挥向量化的作用，每一次会同时对 16 个 6x6 的 tile 矩阵进行操作，因为每个 AVX512 寄存器能存储 16 个单精度浮点数。通过循环展开手写 AVX 指令的方式，把 16 个矩阵中的数读取出来并重新排列，之后再批量与矩阵 G 相乘，相乘操作也全部展开用 AVX 指令的方式实现。这样批量操作可以充分发挥单核 SIMD 指令的效能，并且每个计算单元不会影响其他部分的内存，不会出现伪共享问题，可以实现完全的并行操作。大量的循环展开虽然增加了代码量，但有利于减少循环开销、也有利于指令流水线的调度。对于边长不满足被步长整除（步长为 4/64，即无法被 4/64 整除）的参数，可以进行 padding。

2. Filter transform

因为卷积核是固定的 3x3 大小，SIMD 无法发挥作用，因此这里使用了循环展开和多线程进行计算卷积核与矩阵 G 相乘，因为卷积核之间也是互相独立的，因此可以尽可能的增加并行度加快运算。最后的赋值也使用 #pragma unroll 指示编译器进行循环展开。

3. Matrix Multiplication

此处主要是两个大矩阵做乘法的工作。优化到这里，我们已经意识到很多地方我们写得再好也不如专业的数学库好，他们甚至可能是用汇编从下往上写起的，进而可能实现了很多我们根本不可能做到的优化。

为此，我们调研并选择了 Intel 的 MKL 数据库，相信他们能够将自己家的硬件性能压榨到极致。经过测试发现，使用 MKL 后有较大的提升（约 **5** 倍），因此这里全部替换成了 mkl 的 sgemm 函数。这里的并行度有两个参数，一个是 36，因为经过 Input transform 得到的三维矩阵有一维是 36，且彼此之间互不影响，可以独立并行。另一个参数是 batch，因为之前把 batch 中所有图像拼在了一起，这里也可以并行，相当于对每个图像做计算。

4. Output transform

与 Input transform 类似，也是对 16 个 6x6 的 tile 矩阵进行操作，最终与矩阵 A 相乘，得到 16 个 4x4 的矩阵，并保存到结果矩阵中。这里也大量使用 AVX 指令进行循环展开，并在函数外层使用 openmp 多线程对不同区域进行操作。

这一步的优化包括算法优化、指令集优化，并且引入了 MKL 替代原来的大矩阵乘法。

在算法实现方面注意调整内存的分布，尽量提高缓存命中；尽量使用循环展开，用代码量和可读性换性能；通过分 tile、分块操作的方法，将算法分割成一个个独立的部分，在保证正确性的前提下尽量提高并行度。这些优化提升了 13 倍左右的性能。 **(170GFlops->2200GFlops)**

### 测试结果

最终，我们的编译指令如下：
```
gcc -std=c11 -fopenmp -lmkl_rt -O3 driver.c winograd.c -o winograd -mavx512f
```

测试 realworld.conf 的测试性能为 **2202GFlops**。

### 参考资料
* [SIMD 简介](https://zhuanlan.zhihu.com/p/55327037?utm_source=wechat_session&utm_medium=social&utm_oi=749557775620665344&utm_campaign=shareopn)
* [AVX / AVX2 指令编程](https://zhuanlan.zhihu.com/p/94649418?utm_source=wechat_session&utm_medium=social&utm_oi=749557775620665344&utm_campaign=shareopn)
* [为什么向量化计算会这么快？](https://zhuanlan.zhihu.com/p/72953129)
* [高性能深度学习的编译优化](https://zhuanlan.zhihu.com/p/390801790?utm_source=wechat_session&utm_medium=social&utm_oi=749557775620665344&utm_campaign=shareopn)
* [卷积神经网络中的 Winograd 快速卷积算法](https://www.cnblogs.com/shine-lee/p/10906535.html)
* [优化 CPU 矩阵乘法](http://yuenshome.space/timeline/2018-12/optimize-cpu-gemm/)
* [矩阵运算库在性能上区别大吗？](https://www.zhihu.com/question/27872849/answer/75968482?utm_source=wechat_session&utm_medium=social&utm_oi=749557775620665344&utm_content=group3_Answer&utm_campaign=shareopn)
* [NCNN](https://github.com/Tencent/ncnn/tree/master/src/layer/arm)
* [FeatherCNN](https://github.com/Tencent/FeatherCNN/blob/booster/src/booster/arm/winograd_kernels_F63.cpp)
* [Optimization of Spatial Convolution in ConvNets on Intel KNL](https://github.com/chasingegg/Winconv/blob/master/reference/Optimization.pdf)
* [Winconv](https://github.com/chasingegg/Winconv/tree/master/winconv_4x3)
* [Fast Algorithms for Convolutional Neural Networks](https://arxiv.org/pdf/1509.09308v2.pdf)
* [Sparse Winograd Convolutional neural networks on small-scale systolic arrays](https://www.researchgate.net/publication/328091476_Sparse_Winograd_Convolutional_neural_networks_on_small-scale_systolic_arrays)
* [Perf](https://www.brendangregg.com/perf.html)

## H5bench

### 应用特性分析

#### 影响 IO 的因素

测量的精度是控制的上限，在 I\O 相关的问题中，如果不能准确地测量性能，就难以进行真正的改进，无法准确地分析瓶颈。因此我们对 Linux I\O 进行了一定的调研，分析了可能影响带宽的因素、如何测量带宽和性能抖动的原因。从这些调研出发，也能引出我们进行优化的主要方向。

● Page Cache
在 Linux 的实现中，文件 Cache 分为 Page Cache 和 Buffer Cache，一个 Page Cache 包含若干个 Buffer Cache。Page Cache 主要用来作为文件系统上的文件数据的缓存来用，Buffer Cache 则主要是设计用来在系统对块设备进行读写的时候，对块进行数据缓存的系统来使用。
当 Linux 内核开始一个读操作，比如，执行 file->f_op->read() 或 file->f_op->write()，它会首先检查需要的数据是否在 Page Cache 中。如果命中 cache，则可直接从内存中读取。因为访问磁盘的速度要远远低于访问内存的速度（ms 和 ns 的差距），因此命中 cache 可以极大地提高 I/O 速度。
具体到 Linux 内核的读操作中，内核首先调用 find_get_page() 尝试在 page cache 中找到需要的数据。如果搜索的页并不在 page cache 中，find_get_page() 返回 NULL，并且内核将分配一个新页面，然后调用 add_to_page_cache_lru() 将之前搜索的页加入 page cache 中。最后调用 readpage() 将需要的数据从磁盘读入。
在对读取同一个文件的带宽进行多次测量时，需要运行
echo 3 >/proc/sys/vm/drop_caches
来手动清空 Page Cache，否则内核存在直接从内存中读取数据的可能性，导致实测的 I/O 带宽失准。

● 异步 I/O
异步 I/O 的工作方式是当进程读写文件时，一旦读写操作进入队列函数就结束，甚至有可能真正的 I/O 数据传输还没有开始，这样调用进程就可以在数据正在传输时继续自己的运行。
io_uring 是 2019 年 Linux5.1 内核引入的取代传统 Linux AIO 的高性能异步 I/O 框架。通过全新的设计，共享内存，IO 过程不需要系统调用，由内核完成 IO 的提交， 以及 IO completion polling 机制，实现了高 IOPS 和高 Bandwidth。
h5bench 提供了使用 HDF5 Asynchronous I/O VOL connector 为支撑的异步 I/O 模式。HDF5 Asynchronous I/O VOL connector 利用 HDF5 异步接口，通过对 I/O 操作尽早调度以及与计算和通信并行执行，有效提高了 HDF5 的整体运行效率。

● 并行 I/O
并行 I/O 是一种使用不同进程同时访问磁盘数据的技术，能够最大化带宽和速度。从 Linux 内核的角度看，块 I/O 操作的基本容器由 bio 结构体表示，其指向一个 bio_vec 结构体链表，这些结构体描述了每隔片段再物理页中的实际位置。块 I/O 层通过 bi_idx 域跟踪块 I/O 操作的完成进度，同时也起到分割 bio 结构体的作用，为 I/O 操作的并行执行提供条件。HDF5 使用 MPI-I/O 的接口，从而支持了部分并行 I/O 操作，提供了比 MPI-I/O 更高级的数据抽象。

● 文件系统 (RAID/NFS)
RAID 通过将多块独立的磁盘（物理硬盘）按不同方式组合起来形成一个磁盘组（逻辑硬盘），从而提供比单个硬盘更高的 I/O 性能和数据冗余。不同的 RAID 级别有不同的 I/O 性能特点。如 RAID5 采用磁盘分段加奇偶校验技术，读出效率很高，写入效率一般。
若实际的文件系统为 NFS，或是包含 NFS 的混合形态，那么实际的 I/O 带宽可能受网络带宽限制。通过网络传输的数据也需要的额外的校验操作，从而进一步限制 I/O 带宽。

● 性能抖动
SSD 内部的 I/O 性能抖动：比特错误（随着使用时间增加，比特错误率越来越高，因比特错误导致的读延迟概率和时间也会随之增高）、读写/擦除操作冲突（当读写操作与擦除操作冲突会导致读写操作延迟）、垃圾回收（SSD 内部进行 GC 时会占用磁盘资源）、读平衡（数据迁移与用户 I/O 之间的资源竞争）。
HDD 内部的 I/O 性能抖动：机械硬盘性能受数据随机/顺序读写因素影响较大。此外，跨盘符时会出现性能抖动。
磁盘带宽占用：当其他进程当前任务竞争磁盘 I/O 时，会出现性能抖动。
内存占用：当系统中存在内存不足的压力时，Linux 内核会利用 flusher 线程对 Page Cache 进行回写动作，势必会影响 I/O 性能，造成性能抖动。

#### 如何测得正确的带宽

● I/O 测试/监控工具
FIO 是一个可以产生很多线程或进程并执行用户指定的特定类型 I/O 操作的工具。FIO 支持十几种不同类型的 io 引擎（libaio、sync、mmap、posixaio、network 等等），可以测试块设备或文件，可以通过多线程或进程模拟各种 io 操作，可以测试统计 iops、带宽和时延等性能。通过 FIO 相应模式测得的设备 I/O 带宽是我们可达到的带宽性能的极限。
系统设备 IO 负载情况的监控工具有 iotop、iostat 等。

### 优化思路

### 优化实现及工程细节
async 分块 压缩 多进程

### 测试结果

### 参考资料

## 总结
