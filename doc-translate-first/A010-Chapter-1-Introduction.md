> Chapter 1. Introduction
~~~
1 장. 소개
~~~

>1.1 From Graphics Processing to General-Purpose Parallel Computing 
~~~
1.1 그래픽 처리에서 범용 병렬 컴퓨팅에 이르기까지
~~~

Driven by the insatiable market demand for realtime, high-definition 3D graphics, the programmable Graphic Processor Unit or GPU has evolved into a highly parallel, multithreaded, manycore processor with tremendous computational horsepower and very high memory bandwidth, as illustrated by Figure 1-1.


    실시간 끊임없는 시장 수요에 힘입어, 고화질 3D 그래픽, 프로그래밍 가능 그래픽 프로세서 유닛이나 GPU는 그림 1-1과 같이, 엄청난 연산 능력과 매우 높은 메모리 대역폭을 가진 고도의 병렬, 멀티스레드, 매니코어 프로세서로 발전했습니다.


2 CUDA C Programming Guide Version 4.2 
2 CUDA C 프로그래밍 가이드 버전 4.2 
 
Figure 1-1. Floating-Point Operations per Second and Memory Bandwidth for the CPU and GPU   
그림 1-1. CPU 및 GPU의 초당 부동 소수점 연산 및 메모리 대역폭

CUDA C Programming Guide Version 4.2 3 
CUDA C 프로그래밍 가이드 버전 4.2 3 
 
The reason behind the discrepancy in floating-point capability between the CPU and the GPU is that the GPU is specialized for compute-intensive, highly parallel computation – exactly what graphics rendering is about – and therefore designed such that more transistors are devoted to data processing rather than data caching and flow control, as schematically illustrated by Figure 1-2.  
CPU와 GPU 간의 부동 소수점 기능의 불일치에 대한 이유는 GPU가 연산 집약형이고 고도의 병렬 계산 (정확히 그래픽 렌더링 같은)에 특화되어 있어서, 더 많은 트랜지스터가 데이터 캐싱이나 흐름 제어보다는 데이터 처리에만 사용되도록 설계되어 있다는 것입니다(그림 1-2에 개략적으로 설명되어 있습니다).
 
Figure 1-2. The GPU Devotes More Transistors to Data Processing 
그림 1-2. GPU는 데이터 처리에 더 많은 트랜지스터를 사용합니다.

More specifically, the GPU is especially well-suited to address problems that can be expressed as data-parallel computations – the same program is executed on many data elements in parallel – with high arithmetic intensity – the ratio of arithmetic operations to memory operations.  
구체적으로, GPU는 데이터 병렬 계산(같은 프로그램이 여러 데이터 요소에서 병렬로 실행됨)과 높은 산술 강도 (산술 연산과 메모리 연산의 비율)로 표현할 수 있는 문제를 해결하는 데 특히 적합합니다.

Because the same program is executed for each data element, there is a lower requirement for sophisticated flow control, and because it is executed on many data elements and has high arithmetic intensity, the memory access latency can be hidden with calculations instead of big data caches. 
동일한 프로그램이 각 데이터 요소에 실행되기 때문에 정교한 흐름 제어에 대한 요구 사항이 낮으며 많은 데이터 요소에서 실행되고 높은 산술 강도를 가지기 때문에 큰 데이터 캐시 대신 계산을 통해 메모리 액세스 대기 시간을 숨길 수 있습니다.

Data-parallel processing maps data elements to parallel processing threads. 
데이터 병렬 처리는 데이터 요소를 병렬 처리 스레드에 매핑합니다.

Many applications that process large data sets can use a data-parallel programming model to speed up the computations. 
대형 데이터 세트를 처리하는 많은 애플리케이션은 데이터 병렬 프로그래밍 모델을 사용하여 계산 속도를 높일 수 있습니다.

In 3D rendering, large sets of pixels and vertices are mapped to parallel threads. 
3D 렌더링에서 픽셀과 꼭지점의 큰 세트는 병렬 스레드에 매핑됩니다.

Similarly, image and media processing applications such as post-processing of rendered images, video encoding and decoding, image scaling, stereo vision, and pattern recognition can map image blocks and pixels to parallel processing threads. 
마찬가지로 렌더링된 이미지의 후 처리, 비디오 인코딩 및 디코딩, 이미지 스케일링, 스테레오 비전 및 패턴 인식과 같은 이미지와 미디어 처리 애플리케이션은 이미지 블록과 픽셀을 병렬 처리 스레드로 매핑할 수 있습니다.

In fact, many algorithms outside the field of image rendering and processing are accelerated by data-parallel processing, from general signal processing or physics simulation to computational finance or computational biology. 
실제로 이미지 렌더링 및 처리 분야 이외의 많은 알고리즘은 일반 신호 처리나 물리 시뮬레이션에서 전산 금융 또는 전산 생물학에 이르기까지 데이터 병렬 처리로 가속화됩니다.

1.2 CUDA™: a General-Purpose Parallel Computing Architecture 
1.2 CUDA™: 범용 병렬 컴퓨팅 아키텍처

In November 2006, NVIDIA introduced CUDA™, a general purpose parallel computing architecture – with a new parallel programming model and instruction set architecture – that leverages the parallel compute engine in NVIDIA GPUs to solve many complex computational problems in a more efficient way than on a CPU. 
2006년 11월에, NVIDIA는 범용 병렬 컴퓨팅 아키텍처인 CUDA™을 새로운 병렬 프로그래밍 모델 및 명령어 세트 아키텍처와 함께 도입하여 NVIDIA GPU의 병렬 컴퓨팅 엔진을 활용하여 CPU보다 더 효율적인 방법으로 복잡한 컴퓨팅 문제를 해결합니다.

CUDA comes with a software environment that allows developers to use C as a high-level programming language. 
CUDA는 개발자가 C를 하이 레벨 프로그래밍 언어로 사용할 수 있는 소프트웨어 환경을 제공합니다.

As illustrated by Figure 1-3, other languages, application programming interfaces, or directives-based approaches are supported, such as FORTRAN, DirectCompute, OpenCL, OpenACC.
그림 1-3에서 볼 수 있듯이, FORTRAN, DirectCompute, OpenCL, OpenACC와 같은 다른 언어, 애플리케이션 프로그래밍 인터페이스 또는 지시문 기반 접근이 지원됩니다.

Figure 1-3. CUDA is Designed to Support Various Languages and Application Programming Interfaces. 
그림 1-3. CUDA는 다양한 언어 및 애플리케이션 인터페이스를 지원하도록 설계되었습니다.

1.3 A Scalable Programming Model 
1.3 확장 가능한 프로그래밍 모델

The advent of multicore CPUs and manycore GPUs means that mainstream processor chips are now parallel systems. 
멀티코어 CPU와 매니코어 GPU의 출현은 메인스트림 프로세서 칩이 현재 병렬 시스템임을 의미합니다.

Furthermore, their parallelism continues to scale with Moore’s law. 
또한 무어의 법칙에 따라 병렬 처리가 계속 확장됩니다.

The challenge is to develop application software that transparently scales its parallelism to leverage the increasing number of processor cores, much as 3D graphics applications transparently scale their parallelism to manycore GPUs with widely varying numbers of cores. 
3D 그래픽 애플리케이션이 다양한 수의 코어를 사용하는 매니코어 GPU에 대한 병렬 처리를 확장할 때와 마찬가지로 증가하는 수의 프로세서 코어를 활용하기 위해 병렬 처리를 투명하게 확장하는 애플리케이션 소프트웨어를 개발하는 것이 도전 과제입니다.

The CUDA parallel programming model is designed to overcome this challenge while maintaining a low learning curve for programmers familiar with standard programming languages such as C. 
CUDA 병렬 프로그래밍 모델은 C와 같은 표준 프로그래밍 언어에 익숙한 프로그래머에게 낮은 학습 곡선을 유지하면서 이러한 도전과제를 극복하도록 설계되었습니다.

At its core are three key abstractions – a hierarchy of thread groups, shared memories, and barrier synchronization – that are simply exposed to the programmer as a minimal set of language extensions. 
스레드 그룹, 공유 메모리 및 배리어 동기화의 계층 구조라는 핵심에는 세 가지 주요 추상화가 있습니다. 이 추상화는 최소한의 언어 확장 집합으로 프로그래머에게 쉽게 노출됩니다.

These abstractions provide fine-grained data parallelism and thread parallelism, nested within coarse-grained data parallelism and task parallelism. 
이러한 추상화는 세분화된 데이터 병렬 처리 및 스레드 병렬 처리를 제공하며 대단위 데이터 병렬 처리 및 작업 병렬 처리 내에서 중첩됩니다.

They guide the programmer to partition the problem into coarse sub-problems that can be solved independently in parallel by blocks of threads, and each sub-problem into finer pieces that can be solved cooperatively in parallel by all threads within the block. 
이들은 프로그래머가 문제를 스레드의 블록과 병행하여 독립적으로 해결될 수 있는 하위 문제로 나눌 수 있으며, 각 하위 문제는 블록 내의 모든 스레드와 병행하여 협력적으로 해결될 수 있는 보다 세밀한 부분으로 안내합니다.

This decomposition preserves language expressivity by allowing threads to cooperate when solving each sub-problem, and at the same time enables automatic scalability. 
이 분할은 각 하위 문제를 해결할 때 스레드가 협력할 수 있게 함으로써 언어 표현을 유지함과 동시에 자동 확장성을 가능하게 합니다.

Indeed, each block of threads can be scheduled on any of the available multiprocessors within a GPU, in any order, concurrently or sequentially, so that a compiled CUDA program can execute on any number of multiprocessors as illustrated by Figure 1-4, and only the runtime system needs to know the physical multiprocessor count. 
실제로 스레드의 각 블록은 GPU 내의 사용가능한 다중 프로세서 중 임의의 순서로, 동시에 또는 순차적으로 예약될 수 있으므로 컴파일된 CUDA 프로그램은 그림 1-4에서 설명한 것처럼 다중 프로세서에서 실행할 수 있습니다. 런타임 시스템은 물리적 다중 프로세서 수를 알아야 합니다.

This scalable programming model allows the CUDA architecture to span a wide market range by simply scaling the number of multiprocessors and memory partitions: from the high-performance enthusiast GeForce GPUs and professional Quadro and Tesla computing products to a variety of inexpensive, mainstream GeForce GPUs (see Appendix A for a list of all CUDA-enabled GPUs). 
이 확장형 프로그래밍 모델을 통해 CUDA 아키텍처는 고성능 광 GeForce GPU, 전문 쿼드로 및 테슬라 컴퓨팅 제품에서 저렴한 종류, 메인스트림 GeForce GPU에 이르기까지 멀티 프로세서 및 메모리 파티션의 수를 확장하여 넓은 시장 범위를 확장할 수 있습니다. 모든 CUDA 지원 GPU 목록은 부록 A를 참조하십시오).
 
A GPU is built around an array of Streaming Multiprocessors (SMs) (see Chapter 4 for more details). 
GPU는 스트리밍 멀티 프로세서 (Streaming Multiprocessor, SM)의 배열 주위에 구축됩니다 (자세한 내용은 4 장 참조).

A multithreaded program is partitioned into blocks of threads that execute independently from each other, so that a GPU with more multiprocessors will automatically execute the program in less time than a GPU with fewer multiprocessors. 
다중 스레드 프로그램은 서로 독립적으로 실행되는 스레드 블록으로 분할되므로 더 많은 다중 프로세서를 갖춘 GPU는 적은 다중프로세서로 더 적은 시간에 프로그램을 자동으로 실행할 수 있습니다.
 
1.4 Document’s Structure 
1.4 문서의 구조

This document is organized into the following chapters:  Chapter 1 is a general introduction to CUDA.  
이 문서는 다음 장으로 구성되어 있습니다: 1 장은 CUDA에 대한 일반적인 소개입니다.

Chapter 2 outlines the CUDA programming model.  
2 장에서는 CUDA 프로그래밍 모델에 대해 간략히 설명합니다. 

Chapter 3 describes the programming interface.  
3 장에서는 프로그래밍 인터페이스에 대해 설명합니다.

Chapter 4 describes the hardware implementation.  
4 장에서는 하드웨어 구현에 대해 설명합니다.

Chapter 5 gives some guidance on how to achieve maximum performance.  
5 장에서는 최대 성능을 얻는 방법에 대한 지침을 제공합니다.

Appendix A lists all CUDA-enabled devices.  
부록 A는 모든 CUDA 지원 장치를 나열합니다.

Appendix B is a detailed description of all extensions to the C language.  
부록 B는 C 언어의 모든 확장에 대한 자세한 설명입니다.

Appendix C lists the mathematical functions supported in CUDA.  
부록 C는 CUDA에서 지원되는 수학 함수를 나열합니다.

Appendix D lists the C++ features supported in device code.  
부록 D는 장치 코드에서 지원되는 C ++ 기능을 나열합니다.

Appendix E gives more details on texture fetching.  
부록 E에서는 텍스처 가져오기에 대해 자세히 설명합니다.

Appendix F gives the technical specifications of various devices, as well as more architectural details.  
부록 F는 다양한 장치의 기술 사양과 자세한 아키텍처 세부 정보를 제공합니다.

Appendix G introduces the low-level driver API. 
부록 G는 하위 레벨 드라이버 API를 소개합니다. 
