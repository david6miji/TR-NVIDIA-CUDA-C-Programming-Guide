# NVIDIA CUDA C Programming Guide

> Version 4.2  - 4/16/2012

* 번역 : NVIDIA CUDA C Programming Guide  
* https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/CUDA_C_Programming_Guide.pdf

## 번역 및 정리

* 미지(zzing0519@gmail.com)
* 데이비드 유(frog@falinux.com)

## 4.1 버전에서 변경 내용

* Updated Chapter 4, Chapter 5, and Appendix F to include information on devices of compute capability 3.0.
* Replaced each reference to “processor core” with “multiprocessor” in
Section 1.3.
* Replaced Table A-1 by a reference to http://developer.nvidia.com/cuda-gpus.
* Added new Section B.13 on the warp shuffle functions

## Table of Contents

[Chapter 1. Introduction]()  
1.1 From Graphics Processing to General-Purpose Parallel Computing  
1.2 CUDA™: a General-Purpose Parallel Computing Architecture  
1.3 A Scalable Programming Model  
1.4 Document’s Structure   

[Chapter 2. Programming Model]()  
2.1 Kernels  
2.2 Thread Hierarchy  
2.3 Memory Hierarchy  
2.4 Heterogeneous Programming  
2.5 Compute Capability  

[Chapter 3. Programming Interface]()  
3.1 Compilation with NVCC  
3.1.1 Compilation Workflow  
3.1.1.1 Offline Compilation  
3.1.1.2 Just-in-Time Compilation  
3.1.2 Binary Compatibility  
3.1.3 PTX Compatibility  
3.1.4 Application Compatibility  
3.1.5 C/C++ Compatibility  
3.1.6 64-Bit Compatibility  
3.2 CUDA C Runtime  
3.2.1 Initialization  
3.2.2 Device Memory  
3.2.3 Shared Memory  
3.2.4 Page-Locked Host Memory  
3.2.4.1 Portable Memory  
3.2.4.2 Write-Combining Memory  
3.2.4.3 Mapped Memory  
3.2.5 Asynchronous Concurrent Execution  
3.2.5.1 Concurrent Execution between Host and Device  
3.2.5.2 Overlap of Data Transfer and Kernel Execution  
3.2.5.3 Concurrent Kernel Execution  
3.2.5.4 Concurrent Data Transfers  
3.2.5.5 Streams  
3.2.5.6 Events  
3.2.5.7 Synchronous Calls  
3.2.6 Multi-Device System  
3.2.6.1 Device Enumeration  
3.2.6.2 Device Selection  
3.2.6.3 Stream and Event Behavior  
3.2.6.4 Peer-to-Peer Memory Access  
3.2.6.5 Peer-to-Peer Memory Copy  
3.2.7 Unified Virtual Address Space  
3.2.8 Error Checking  
3.2.9 Call Stack  
3.2.10 Texture and Surface Memory  
3.2.10.1 Texture Memory  
3.2.10.2 Surface Memory  
3.2.10.3 CUDA Arrays  
3.2.10.4 Read/Write Coherency  
3.2.11 Graphics Interoperability  
3.2.11.1 OpenGL Interoperability  
3.2.11.2 Direct3D Interoperability  
3.2.11.3 SLI Interoperability  
3.3 Versioning and Compatibility  
3.4 Compute Modes  
3.5 Mode Switches  
3.6 Tesla Compute Cluster Mode for Windows  

[Chapter 4. Hardware Implementation]()  
4.1 SIMT Architecture  
4.2 Hardware Multithreading  

[Chapter 5. Performance Guidelines]()  
5.1 Overall Performance Optimization Strategies  
5.2 Maximize Utilization  
5.2.1 Application Level  
5.2.2 Device Level  
5.2.3 Multiprocessor Level  
5.3 Maximize Memory Throughput  
5.3.1 Data Transfer between Host and Device  
5.3.2 Device Memory Accesses  
5.3.2.1 Global Memory  
5.3.2.2 Local Memory  
5.3.2.3 Shared Memory  
5.3.2.4 Constant Memory  
5.3.2.5 Texture and Surface Memory  
5.4 Maximize Instruction Throughput  
5.4.1 Arithmetic Instructions  
5.4.2 Control Flow Instructions  
5.4.3 Synchronization Instruction  

[Appendix A. CUDA-Enabled GPUs]()  

[Appendix B. C Language Extensions]()  
B.1 Function Type Qualifiers  
B.1.1 __device__  
B.1.2 __global__  
B.1.3 __host__  
B.1.4 __noinline__ and __forceinline__  
B.2 Variable Type Qualifiers  
B.2.1 __device__  
B.2.2 __constant__  
B.2.3 __shared__  
B.2.4 __restrict__  
B.3 Built-in Vector Types  
B.3.1 char1, uchar1, char2, uchar2, char3, uchar3, char4, uchar4, short1,
ushort1, short2, ushort2, short3, ushort3, short4, ushort4, int1, uint1, int2, uint2,
int3, uint3, int4, uint4, long1, ulong1, long2, ulong2, long3, ulong3, long4, ulong4,
longlong1, ulonglong1, longlong2, ulonglong2, float1, float2, float3, float4, double1,
double2  
B.3.2 dim3  
B.4 Built-in Variables   
B.4.1 gridDim  
B.4.2 blockIdx   
B.4.3 blockDim  
B.4.4 threadIdx  
B.4.5 warpSize  
B.5 Memory Fence Functions  
B.6 Synchronization Functions  
B.7 Mathematical Functions  
B.8 Texture Functions  
B.8.1 tex1Dfetch()  
B.8.2 tex1D()  
B.8.3 tex2D()  
B.8.4 tex3D()  
B.8.5 tex1DLayered()  
B.8.6 tex2DLayered()  
B.8.7 texCubemap()  
B.8.8 texCubemapLayered()  
B.8.9 tex2Dgather()  
B.9 Surface Functions  
B.9.1 surf1Dread()  
B.9.2 surf1Dwrite()  
B.9.3 surf2Dread()  
B.9.4 surf2Dwrite()  
B.9.5 surf3Dread()  
B.9.6 surf3Dwrite()  
B.9.7 surf1DLayeredread()  
B.9.8 surf1DLayeredwrite()   
B.9.9 surf2DLayeredread()  
B.9.10 surf2DLayeredwrite()  
B.9.11 surfCubemapread()  
B.9.12 surfCubemapwrite()  
B.9.13 surfCubemabLayeredread()  
B.9.14 surfCubemapLayeredwrite()  
B.10 Time Function  
B.11 Atomic Functions  
B.11.1 Arithmetic Functions  
B.11.1.1 atomicAdd()  
B.11.1.2 atomicSub()  
B.11.1.3 atomicExch()  
B.11.1.4 atomicMin()  
B.11.1.5 atomicMax()  
B.11.1.6 atomicInc()  
B.11.1.7 atomicDec()  
B.11.1.8 atomicCAS()  
B.11.2 Bitwise Functions  
B.11.2.1 atomicAnd()  
B.11.2.2 atomicOr()  
B.11.2.3 atomicXor()  
B.12 Warp Vote Functions  
B.13 Warp Shuffle Functions  
B.13.1 Synopsys  
B.13.2 Description  
B.13.3 Return Value  
B.13.4 Notes  
B.13.5 Examples  
B.13.5.1 Broadcast of a single value across a warp  
B.13.5.2 Inclusive plus-scan across sub-partitions of 8 threads  
B.13.5.3 Reduction across a warp  
B.14 Profiler Counter Function  
B.15 Assertion  
B.16 Formatted Output  
B.16.1 Format Specifiers  
B.16.2 Limitations  
B.16.3 Associated Host-Side API  
B.16.4 Examples  
B.17 Dynamic Global Memory Allocation  
B.17.1 Heap Memory Allocation  
B.17.2 Interoperability with Host Memory API  
B.17.3 Examples  
B.17.3.1 Per Thread Allocation  
B.17.3.2 Per Thread Block Allocation  
B.17.3.3 Allocation Persisting Between Kernel Launches  
B.18 Execution Configuration  
B.19 Launch Bounds  
B.20 #pragma unroll  

[Appendix C. Mathematical Functions]()  
C.1 Standard Functions  
C.1.1 Single-Precision Floating-Point Functions  
C.1.2 Double-Precision Floating-Point Functions  
C.2 Intrinsic Functions  
C.2.1 Single-Precision Floating-Point Functions  
C.2.2 Double-Precision Floating-Point Functions  

[Appendix D. C/C++ Language Support]()  
D.1 Code Samples  
D.1.1 Data Aggregation Class  
D.1.2 Derived Class  
D.1.3 Class Template  
D.1.4 Function Template  
D.1.5 Functor Class  
D.2 Restrictions  
D.2.1 Qualifiers  
D.2.1.1 Device Memory Qualifiers  
D.2.1.2 Volatile Qualifier  
D.2.2 Pointers  
D.2.3 Operators  
D.2.3.1 Assignment Operator  
D.2.3.2 Address Operator  
D.2.4 Functions  
D.2.4.1 Function Parameters  
D.2.4.2 Static Variables within Function  
D.2.4.3 Function Pointers  
D.2.4.4 Function Recursion  
D.2.5 Classes  
D.2.5.1 Data Members  
D.2.5.2 Function Members  
D.2.5.3 Constructors and Destructors  
D.2.5.4 Virtual Functions  
D.2.5.5 Virtual Base Classes  
D.2.5.6 Windows-Specific  
D.2.6 Templates  

[Appendix E. Texture Fetching]()  
E.1 Nearest-Point Sampling  
E.2 Linear Filtering  
E.3 Table Lookup  

[Appendix F. Compute Capabilities]()  
F.1 Features and Technical Specifications  
F.2 Floating-Point Standard  
F.3 Compute Capability 1.x  
F.3.1 Architecture  
F.3.2 Global Memory  
F.3.2.1 Devices of Compute Capability 1.0 and 1.1  
F.3.2.2 Devices of Compute Capability 1.2 and 1.3  
F.3.3 Shared Memory  
F.3.3.1 32-Bit Strided Access  
F.3.3.2 32-Bit Broadcast Access  
F.3.3.3 8-Bit and 16-Bit Access  
F.3.3.4 Larger Than 32-Bit Access  
F.4 Compute Capability 2.x  
F.4.1 Architecture 
F.4.2 Global Memory  
F.4.3 Shared Memory  
F.4.3.1 32-Bit Strided Access  
F.4.3.2 Larger Than 32-Bit Access  
F.4.4 Constant Memory  
F.5 Compute Capability 3.0  
F.5.1 Architecture  
F.5.2 Global Memory  
F.5.3 Shared Memory  
F.5.3.1 64-Bit Mode  
F.5.3.2 32-Bit Mode  

[Appendix G. Driver API]()  
G.1 Context  
G.2 Module   
G.3 Kernel Execution  
G.4 Interoperability between Runtime and Driver APIs

