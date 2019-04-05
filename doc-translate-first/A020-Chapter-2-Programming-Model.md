# Chapter 2. Programming Model 
> 2 장. 프로그래밍 모델

This chapter introduces the main concepts behind the CUDA programming model by outlining how they are exposed in C. 
> 이 장에서는 C로 노출되는 방법을 개괄하여 CUDA 프로그래밍 모델의 기본 개념을 소개합니다.

An extensive description of CUDA C is given in Chapter 3. 
> CUDA C에 대한 자세한 설명은 3 장에 있습니다.

Full code for the vector addition example used in this chapter and the next can be found in the vectorAdd SDK code sample. 
> 이 장과 다음 장에서 사용되는 벡터 추가 예제의 전체 코드는 vectorAdd SDK 코드 샘플에서 찾을 수 있습니다.

## 2.1 Kernels 
> 2.1 커널

CUDA C extends C by allowing the programmer to define C functions, called kernels, that, when called, are executed N times in parallel by N different CUDA threads, as opposed to only once like regular C functions. 
> CUDA C는 프로그래머가 커널이라고하는 C 함수를 정의할 수 있게 하여 C를 확장합니다. 이 함수는 호출시, 일반 C 함수처럼 한 번만 반대되므로 N 개의 다른 CUDA 스레드에 의해 병렬로 N번 실행됩니다.

A kernel is defined using the __global__ declaration specifier and the number of CUDA threads that execute that kernel for a given kernel call is specified using a new <<<…>>> execution configuration syntax (see Appendix B.18). 
> 커널은 __global__ 선언 지정자를 사용하여 정의되며 주어진 커널 호출에 대해 해당 커널을 실행하는 CUDA 스레드의 수는 새로운 <<<...>>> 실행 구성 구문 (부록 B.18 참조)을 사용하여 지정됩니다.

Each thread that executes the kernel is given a unique thread ID that is accessible within the kernel through the built-in threadIdx variable. 
> 커널을 실행하는 각 스레드는 내장 threadIdx 변수를 통해 커널 내에서 액세스할 수 있는 고유한 스레드 ID가 제공됩니다.

As an illustration, the following sample code adds two vectors A and B of size N and stores the result into vector C:  
> 예를 들어, 다음 샘플 코드는 사이즈 N의 두 벡터 A와 B를 더하고 그 결과를 벡터 C에 저장합니다. 

Here, each of the N threads that execute VecAdd() performs one pair-wise addition.  
> 여기서, VecAdd()를 실행하는 N 개의 스레드 각각은 한 쌍의 덧셈을 수행합니다.

## 2.2 Thread Hierarchy 
> 2.2 스레드 계층 구조

For convenience, threadIdx is a 3-component vector, so that threads can be identified using a one-dimensional, two-dimensional, or three-dimensional thread index, forming a one-dimensional, two-dimensional, or three-dimensional thread block. 
> 편의상, threadIdx는 3-성분 벡터이므로 스레드는 1 차원, 2 차원 또는 3 차원 스레드 인덱스를 사용하여 식별될 수 있거나 1 차원, 2 차원 또는 3 차원 스레드 블록을 형성합니다 .

This provides a natural way to invoke computation across the elements in a domain such as a vector, matrix, or volume. 
> 이는 벡터, 매트릭스나 볼륨 같은 도메인의 요소를 통해 계산을 호출하는 자연스러운 방법을 제공합니다.

The index of a thread and its thread ID relate to each other in a straightforward way: 
> 스레드의 색인과 스레드 ID는 서로 직접적인 관련이 있습니다.

For a one-dimensional block, they are the same; for a two-dimensional block of size (Dx, Dy), the thread ID of a thread of index (x, y) is (x + y Dx); for a three-dimensional block of size (Dx, Dy, Dz), the thread ID of a thread of index (x, y, z) is (x + y Dx + z Dx Dy). 
> 1 차원 블록의 경우, 이들은 동일합니다. 사이즈 (Dx, Dy)의 2차원 블록의 경우, 인덱스 (x, y) 스레드의 스레드 ID는 (x + yDx)이고; 사이즈(Dx, Dy, Dz)의 3차원 블록의 경우 인덱스(x, y, z) 스레드의 스레드 ID는 (x + y Dx + z Dx Dy)입니다.

As an example, the following code adds two matrices A and B of size NxN and stores the result into matrix C:  
> 예를 들어, 다음 코드는 사이즈 NxN의 두 매트릭스 A와 B를 더하고 그 결과를 매트릭스 C에 저장합니다.

There is a limit to the number of threads per block, since all threads of a block are expected to reside on the same processor core and must share the limited memory resources of that core. 
> 블록의 모든 스레드가 동일한 프로세서 코어에 상주할 것으로 예상되고 해당 코어의 제한된 메모리 자원을 공유해야 하기 때문에 블록 당 스레드 수에는 제한이 있습니다.

On current GPUs, a thread block may contain up to 1024 threads. 
> 현재 GPU에서 스레드 블록은 최대 1024개의 스레드를 포함할 수 있습니다.

However, a kernel can be executed by multiple equally-shaped thread blocks, so that the total number of threads is equal to the number of threads per block times the number of blocks. 
> 그러나 커널은 여러 개의 같은 모양의 스레드 블록으로 실행될 수 있으므로 총 스레드 수는 블록 당 스레드 수와 블록 수를 곱한 값과 같습니다.

Blocks are organized into a one-dimensional, two-dimensional, or three-dimensional grid of thread blocks as illustrated by Figure 2-1. 
> 블록은 그림 2-1에 표시된 것처럼 스레드 블록의 1차원, 2차원 또는 3차원 그리드로 구성됩니다.

The number of thread blocks in a grid is usually dictated by the size of the data being processed or the number of processors in the system, which it can greatly exceed. 
> 그리드의 스레드 블록 수는 일반적으로 처리되는 데이터의 크기 또는 시스템의 프로세서 수에 따라 크게 달라질 수 있습니다.
 
Figure 2-1. Grid of Thread Blocks 
> 그림 2-1. 스레드 블록의 그리드
 
The number of threads per block and the number of blocks per grid specified in the <<<…>>> syntax can be of type int or dim3.   
> <<<...>>> 구문에 지정된 블록 당 스레드 수와 그리드 당 블록 수는 int나 dim3 유형이 될 수 있습니다.

Two-dimensional blocks or grids can be specified as in the example above. 
> 위의 예에서와 같이 2차원 블록 또는 그리드를 지정할 수 있습니다.

Each block within the grid can be identified by a one-dimensional, two-dimensional, or three-dimensional index accessible within the kernel through the built-in blockIdx variable. 
> 그리드 내의 각 블록은 내장된 blockIdx 변수를 통해 커널 내에서 액세스할 수있는 1 차원, 2 차원 또는 3 차원 인덱스로 식별할 수 있습니다.

The dimension of the thread block is accessible within the kernel through the built-in blockDim variable. 
> 스레드 블록의 차원은 내장된 blockDim 변수를 통해 커널 내에서 액세스할 수 있습니다.

Extending the previous MatAdd() example to handle multiple blocks, the code becomes as follows.
> 이전 MatAdd() 예제를 확장하여 다중 블록을 처리하면 코드는 다음과 같이 됩니다.

A thread block size of 16x16 (256 threads), although arbitrary in this case, is a common choice. 
> 이 경우 임의이지만 스레드 블록 크기가 16x16 (256 스레드)인 것은 일반적인 선택입니다.

The grid is created with enough blocks to have one thread per matrix element as before. 
> 그리드는 이전과 같이 매트릭스 요소 당 하나의 스레드를 갖도록 충분한 블록으로 만들어집니다.

For simplicity, this example assumes that the number of threads per grid in each dimension is evenly divisible by the number of threads per block in that dimension, although that need not be the case. 
> 간단하게, 그럴 필요는 없지만, 이 예제에서는 각 차원의 그리드 당 스레드 수가 해당 차원의 블록 당 스레드 수로 균등하게 나눌 수 있다고 가정합니다.

Thread blocks are required to execute independently: It must be possible to execute them in any order, in parallel or in series. 
> 스레드 블록은 독립적으로 실행해야 합니다. 임의의 순서, 병렬 또는 연속으로 실행할 수 있어야 합니다.

This independence requirement allows thread blocks to be scheduled in any order across any number of cores as illustrated by Figure 1-4, enabling programmers to write code that scales with the number of cores. 
> 이러한 독립성 요구 사항을 통해 그림 1-4에 표시된 것처럼 임의의 수의 코어에서 스레드 블록을 임의의 순서로 스케줄링될 수 있으므로 프로그래머는 코어 수에 따라 확장되는 코드를 작성할 수 있습니다.

Threads within a block can cooperate by sharing data through some shared memory and by synchronizing their execution to coordinate memory accesses. 
> 블록 내의 스레드는 일부 공유 메모리를 통해 데이터를 공유하고 실행을 동기화하여 메모리 액세스를 조정함으로써 협조할 수 있습니다.

More precisely, one can specify synchronization points in the kernel by calling the __syncthreads() intrinsic function; __syncthreads() acts as a barrier at which all threads in the block must wait before any is allowed to proceed. 
> 보다 정확하게는 __syncthreads() 내적 함수를 호출하여 커널에서 동기화 지점을 지정할 수 있습니다. __syncthreads()는 블록의 모든 스레드가 처리를 진행하기 전에 대기해야 하는 장벽 역할을 합니다.

Section 3.2.3 gives an example of using shared memory. 
> 3.2.3 절은 공유 메모리를 사용하는 예제입니다.

For efficient cooperation, the shared memory is expected to be a low-latency memory near each processor core (much like an L1 cache) and __syncthreads() is expected to be lightweight. 
> 효율적인 협업을 위해 공유 메모리는 각 프로세서 코어 (L1 캐시와 유사한) 근처에서 대기 시간이 짧은 메모리로 예상되며 __syncthreads()는 경량이 될 것으로 예상됩니다.

## 2.3 Memory Hierarchy 
> 2.3 메모리 계층 구조

CUDA threads may access data from multiple memory spaces during their execution as illustrated by Figure 2-2. 
> CUDA 스레드는 그림 2-2에 표시된 것처럼 실행되는 동안 다중 메모리 공간의 데이터에 액세스할 수 있습니다.

Each thread has private local memory. 
> 각 스레드는 사적인 로컬 메모리를 가지고 있습니다.

Each thread block has shared memory visible to all threads of the block and with the same lifetime as the block. 
> 각 스레드 블록은 블록의 모든 스레드가 볼 수있는 공유 메모리와 블록과 동일한 수명을 가집니다.

All threads have access to the same global memory. 
> 모든 스레드는 동일한 전역 메모리에 액세스할 수 있습니다.

There are also two additional read-only memory spaces accessible by all threads: the constant and texture memory spaces. 
> 또한 모든 스레드가 액세스할 수 있는 두 개의 추가 읽기 전용 메모리 공간 (상수 및 텍스처 메모리 공간)이 있습니다.

The global, constant, and texture memory spaces are optimized for different memory usages (see Sections 5.3.2.1, 5.3.2.4, and 5.3.2.5). 
> 전역, 상수 및 텍스처 메모리 공간은 다른 메모리 사용법에 맞게 최적화되어 있습니다 (5.3.2.1, 5.3.2.4 및 5.3.2.5절 참조).

Texture memory also offers different addressing modes, as well as data filtering, for some specific data formats (see Section 3.2.10). 
> 또한 텍스처 메모리는 특정 데이터 형식 (3.2.10절 참조)에 대해 데이터 필터링뿐만 아니라 다른 주소 지정 모드를 제공합니다.

The global, constant, and texture memory spaces are persistent across kernel launches by the same application. 
> 전역, 상수 및 텍스처 메모리 공간은 동일한 애플리케이션에서 커널을 시작한 후에도 지속됩니다.

## 2.4 Heterogeneous Programming 
> 2.4 이기종(이질형) 프로그래밍

As illustrated by Figure 2-3, the CUDA programming model assumes that the CUDA threads execute on a physically separate device that operates as a coprocessor to the host running the C program. 
> 그림 2-3에서 볼 수 있듯이, CUDA 프로그래밍 모델은 C 프로그램을 실행하는 호스트에 대한 보조 프로세서로 작동하는 물리적으로 분리된 장치에서 CUDA 스레드가 실행한다고 가정합니다.

This is the case, for example, when the kernels execute on a GPU and the rest of the C program executes on a CPU. 
> 예를 들어, 커널이 GPU에서 실행되고 나머지 C 프로그램이 CPU에서 실행되는 경우입니다. 
 
The CUDA programming model also assumes that both the host and the device maintain their own separate memory spaces in DRAM, referred to as host memory and device memory, respectively. 
> CUDA 프로그래밍 모델은 또한 호스트와 디바이스 모두가 호스트 메모리와 디바이스 메모리라고 하는 각각 별도의 메모리 공간을 DRAM에 유지한다고 가정합니다.

Therefore, a program manages the global, constant, and texture memory spaces visible to kernels through calls to the CUDA runtime (described in Chapter 3). 
> 따라서 프로그램은 CUDA 런타임 (3장에서 설명함) 호출을 통해 커널에 표시되는 전역, 상수 및 텍스처 메모리 공간을 관리합니다.

This includes device memory allocation and deallocation as well as data transfer between host and device memory. 
> 여기에는 호스트 메모리와 디바이스 메모리 간의 데이터 전송은 물론 장치 메모리 할당 및 할당 취소가 포함됩니다.
 
Serial code executes on the host while parallel code executes on the device. 
> 병렬 코드가 장치에서 실행되는 동안 직렬 코드가 호스트에서 실행됩니다.


## 2.5 Compute Capability 
> 2.5 계산 기능

The compute capability of a device is defined by a major revision number and a minor revision number. 
> 장치의 계산 기능은 주요 개정 번호와 부 개정 번호에 의해 정의됩니다.

Devices with the same major revision number are of the same core architecture. 
> 동일한 주요 개정 번호를 가진 장치는 동일한 핵심 아키텍처입니다.

The major revision number is 3 for devices based on the Kepler architecture, 2 for devices based on the Fermi architecture, and 1 for devices based on the Tesla architecture. 
> 주요 개정 번호는 Kepler 아키텍처 기반 장치의 경우 3개, Fermi 아키텍처 기반 장치의 경우 2개, Tesla 아키텍처 기반 장치의 경우 1개입니다.

The minor revision number corresponds to an incremental improvement to the core architecture, possibly including new features. 
> 부 개정 번호는 새로운 기능을 포함하여 핵심 아키텍처에 대한 점진적 개선에 해당합니다.

Appendix A lists of all CUDA-enabled devices along with their compute capability. 
> 부록 A에는 모든 CUDA 지원 디바이스와 해당 컴퓨팅 기능이 나와 있습니다.

Appendix F gives the technical specifications of each compute capability. 
> 부록 F는 각 계산 기능의 기술 명세를 제공합니다.
