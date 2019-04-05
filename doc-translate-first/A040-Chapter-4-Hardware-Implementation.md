# Chapter 4. Hardware Implementation 
> 4 장. 하드웨어 구현

The CUDA architecture is built around a scalable array of multithreaded Streaming Multiprocessors (SMs). 
> CUDA 아키텍처는 멀티 스레드 스트리밍 다중프로세서 (Multipreaded Streaming Multiprocessor, SM)의 확장 가능한 배열을 기반으로 합니다.

When a CUDA program on the host CPU invokes a kernel grid, the blocks of the grid are enumerated and distributed to multiprocessors with available execution capacity. 
> 호스트 CPU의 CUDA 프로그램이 커널 그리드를 호출하면 그리드 블록이 열거되고 사용 가능한 실행 용량이 있는 다중프로세서로 분산됩니다.

The threads of a thread block execute concurrently on one multiprocessor, and multiple thread blocks can execute concurrently on one multiprocessor. 
> 스레드 블록의 스레드는 하나의 다중프로세서에서 동시에 실행되며 다중 스레드 블록은 하나의 다중프로세서에서 동시에 실행할 수 있습니다.

As thread blocks terminate, new blocks are launched on the vacated multiprocessors. 
> 스레드 블록이 종료되면 비워진 다중프로세서에서 새로운 블록이 시작됩니다.

A multiprocessor is designed to execute hundreds of threads concurrently. 
> 다중프로세서는 수백 개의 스레드를 동시에 실행하도록 설계되었습니다.

To manage such a large amount of threads, it employs a unique architecture called SIMT (Single-Instruction, Multiple-Thread) that is described in Section 4.1. 
> 이처럼 많은 양의 스레드를 관리하기 위해 4.1 절에서 설명하는 SIMT (Single-Instruction, Multiple-Thread)라는 고유한 아키텍처를 채택합니다.

The instructions are pipelined to leverage instruction-level parallelism within a single thread, as well as thread-level parallelism extensively through simultaneous hardware multithreading as detailed in Section 4.2. 
> 명령은 섹션 4.2에서 설명하는 것처럼 동시 하드웨어 멀티스레딩을 통해 광범위하게 스레드 수준 병렬 처리뿐만 아니라 단일 스레드 내에서 명령어 수준 병렬 처리를 활용하도록 파이프라인 처리됩니다.

Unlike CPU cores they are issued in order however and there is no branch prediction and no speculative execution.  
> CPU 코어와 달리 이들은 순서대로 발행되지만 분기 예측도 없고 추측 실행도 없습니다.

Sections 4.1 and 4.2 describe the architecture features of the streaming multiprocessor that are common to all devices. 
> 섹션 4.1과 4.2는 모든 디바이스에 공통적인 스트리밍 다중프로세서의 아키텍처 기능을 설명합니다.

Sections F.3.1, F.4.1, and F.5.1 provide the specifics for devices of compute capabilities 1.x, 2.x, and 3.0, respectively. 
> F.3.1, F.4.1 및 F.5.1 절은 각각 컴퓨팅 기능 1.x, 2.x 및 3.0의 디바이스에 대한 세부 사항을 제공합니다.

## 4.1 SIMT Architecture
> 4.1 SIMT 아키텍처
 
The multiprocessor creates, manages, schedules, and executes threads in groups of 32 parallel threads called warps. 
> 다중프로세서는 워프(warps)라고 하는 32개의 병렬 스레드 그룹으로 스레드를 생성, 관리, 예약 및 실행합니다.

Individual threads composing a warp start together at the same program address, but they have their own instruction address counter and register state and are therefore free to branch and execute independently. 
> 워프를 구성하는 개별 스레드는 동일한 프로그램 주소에서 함께 시작하지만 고유한 명령 주소 카운터 및 레지스터 상태를 가지므로 자유롭게 분기하고 독립적으로 실행할 수 있습니다.

The term warp originates from weaving, the first parallel thread technology. A half-warp is either the first or second half of a warp. 
> 워프라는 용어는 첫 번째 병렬 스레드 기술인 제직을 기원으로 합니다. 반-워프는 워프의 첫 번째 또는 두 번째 절반입니다.

A quarter-warp is either the first, second, third, or fourth quarter of a warp. 
> 1/4 워프는 워프의 첫 번째, 두 번째, 세 번째 또는 네 번째 분기 중 하나입니다.

When a multiprocessor is given one or more thread blocks to execute, it partitions them into warps and each warp gets scheduled by a warp scheduler for execution. 
> 다중프로세서가 실행할 하나 이상의 스레드 블록이 주어지면 다중프로세서에 워프를 분할하고 각 워프를 실행하기 위해 워프 스케줄러에 의해 스케줄링합니다.

The way a block is partitioned into warps is always the same; each warp contains threads of consecutive, increasing thread IDs with the first warp containing thread 0. 
> 블록이 워프로 분할되는 방식은 항상 동일합니다. 각각의 워프는 스레드 0을 포함하는 첫 번째 워프와 함께 연속적으로 증가하는 스레드 ID의 스레드를 포함합니다.

Section 2.2 describes how thread IDs relate to thread indices in the block. 
> 2.2 절에서는 스레드 ID가 블록의 스레드 색인과 어떻게 관련되는지 설명합니다.
 
A warp executes one common instruction at a time, so full efficiency is realized when all 32 threads of a warp agree on their execution path. 
> 워프는 한 번에 하나의 공통 명령을 실행하므로 모든 32 스레드의 워프가 실행 경로에서 일치할 때 전체 효율이 실현됩니다.

If threads of a warp diverge via a data-dependent conditional branch, the warp serially executes each branch path taken, disabling threads that are not on that path, and when all paths complete, the threads converge back to the same execution path. 
> 워프의 스레드가 데이터 종속 조건부 분기를 통해 분기되면 워프는 취해진 각 분기 경로를 순차적으로 실행하고 해당 경로에 없는 스레드를 비활성화하며 모든 경로가 완료되면 스레드가 다시 동일한 실행 경로로 수렴합니다.

Branch divergence occurs only within a warp; different warps execute independently regardless of whether they are executing common or disjoint code paths. 
> 브랜치 분기는 워프 내에서만 발생합니다. 다른 워프는 공통 또는 비연속 코드 경로를 실행하는지 여부에 관계없이 독립적으로 실행됩니다.

The SIMT architecture is akin to SIMD (Single Instruction, Multiple Data) vector organizations in that a single instruction controls multiple processing elements. 
> SIMT 아키텍처는 단일 명령어가 여러 프로세싱 요소를 제어한다는 점에서 SIMD (단일 명령어, 다중 데이터) 벡터 조직과 유사합니다.

A key difference is that SIMD vector organizations expose the SIMD width to the software, whereas SIMT instructions specify the execution and branching behavior of a single thread. 
> 중요한 차이점은 SIMD 벡터 조직이 SIMD 너비를 소프트웨어에 공개하는 반면 SIMT 명령어는 단일 스레드의 실행 및 분기 동작을 지정한다는 것입니다.

In contrast with SIMD vector machines, SIMT enables programmers to write thread-level parallel code for independent, scalar threads, as well as data-parallel code for coordinated threads. 
> SIMD 벡터 머신과는 달리 SIMT를 사용하면 프로그래머는 독립적인 스칼라 스레드에 대한 스레드 수준 병렬 코드를 좌표 스레드에 대한 데이터 병렬 코드만큼 잘 작성할 수 있습니다.

For the purposes of correctness, the programmer can essentially ignore the SIMT behavior; however, substantial performance improvements can be realized by taking care that the code seldom requires threads in a warp to diverge. 
> 정확성(정확한 목적)을 위해, 프로그래머는 근본적으로 SIMT 동작을 무시할 수 있지만, 코드가 워프 내의 스레드를 분기할 필요가 거의 없다는 것을 유의함으로써 상당한 성능 향상을 실현할 수 있습니다.

In practice, this is analogous to the role of cache lines in traditional code: 
> 실제로 이는 전통적인 코드에서 캐시 라인의 역할과 유사합니다.

Cache line size can be safely ignored when designing for correctness but must be considered in the code structure when designing for peak performance. 
> 캐시 라인 크기는 정확성을 위해 설계할 때 안전하게 무시할 수 있지만 성능 극대화를 위해 설계할 때는 코드 구조에서 고려해야 합니다.

Vector architectures, on the other hand, require the software to coalesce loads into vectors and manage divergence manually. 
> 반면에 벡터 아키텍처는 소프트웨어가 로드를 벡터로 병합하고 분산을 수동으로 관리할 필요가 있습니다.

If a non-atomic instruction executed by a warp writes to the same location in global or shared memory for more than one of the threads of the warp, the number of serialized writes that occur to that location varies depending on the compute capability of the device (see Sections F.3.2, F.3.3, F.4.2, F.4.3, F.5.2, and F.5.3) and which thread performs the final write is undefined. 
> 워프에 의해 실행되는 비원자 명령어가 워프의 하나 이상의 스레드에 대해 전역 또는 공유 메모리의 동일한 위치에 쓰는 경우, 해당 위치에서 발생하는 직렬화된 쓰기의 수는 디바이스의 컴퓨팅 기능에 따라 다릅니다(F.3.2 절, F.3.3 절, F.4.2 절, F.4.3 절, F.5.2 절, F.5.3 절 참조) 최종 쓰기를 수행하는 쓰레드는 정의되지 않습니다.

If an atomic instruction (see Section B.11) executed by a warp reads, modifies, and writes to the same location in global memory for more than one of the threads of the warp, each read, modify, write to that location occurs and they are all serialized, but the order in which they occur is undefined. 
> 워프에 의해 실행된 원자 명령 (B.11 절 참조)이 워프의 스레드 중 하나 이상에 대해 전역 메모리의 동일한 위치를 읽고, 수정하고, 쓰는 경우, 해당 위치에 대한 읽기, 수정, 쓰기가 발생하고 모두 직렬화되지만 발생 순서는 정의되지 않습니다.

## 4.2 Hardware Multithreading
> 4.2 하드웨어 멀티스레딩
 
The execution context (program counters, registers, etc) for each warp processed by a multiprocessor is maintained on-chip during the entire lifetime of the warp. 
> 다중프로세서에 의해 처리된 각 워프에 대한 실행 컨텍스트 (프로그램 카운터, 레지스터 등)는 워프의 전체 수명 중에 온-칩 (on-chip)으로 관리 유지됩니다.

Therefore, switching from one execution context to another has no cost, and at every instruction issue time, a warp scheduler selects a warp that has threads ready to execute its next instruction (the active threads of the warp) and issues the instruction to those threads. 
> 따라서 하나의 실행 컨텍스트에서 다른 실행 컨텍스트로 전환하는 데는 비용이 들지 않으며 모든 명령 발행 시간에 워프 스케줄러는 다음 명령어 (워프의 활성 스레드)를 실행할 준비가 된 스레드가 있는 워프를 선택하고 해당 스레드에 명령을 발행합니다 .

In particular, each multiprocessor has a set of 32-bit registers that are partitioned among the warps, and a parallel data cache or shared memory that is partitioned among the thread blocks. 
> 특히 각 다중프로세서에는 분할된 32 비트 레지스터 세트가 워프 사이에 있으며 병렬 데이터 캐시나 공유 메모리가 스레드 블록 사이에 있습니다.

The number of blocks and warps that can reside and be processed together on the multiprocessor for a given kernel depends on the amount of registers and shared memory used by the kernel and the amount of registers and shared memory available on the multiprocessor. 
> 주어진 커널의 경우 다중프로세서에서 같이 상주하고 처리될 수 있는 블록 및 워프의 수는 커널이 사용하는 레지스터 및 공유 메모리의 양과 다중프로세서에서 사용할 수 있는 레지스터 및 공유 메모리의 양에 따라 달라집니다.

There are also a maximum number of resident blocks and a maximum number of resident warps per multiprocessor. 
> 다중프로세서 당 상주 블록 최대수 및 상주 워프 최대수도 있습니다.

These limits as well the amount of registers and shared memory available on the multiprocessor are a function of the compute capability of the device and are given in Appendix F. 
> 이러한 제한은 또한 다중프로세서에서 사용할 수 있는 레지스터 및 공유 메모리의 양이 디바이스의 계산 기능 함수이며 부록 F에 나와 있습니다.

If there are not enough registers or shared memory available per multiprocessor to process at least one block, the kernel will fail to launch. 
> 최소 하나의 블록을 처리하기 위해 다중프로세서 당 사용가능한 레지스터 또는 공유 메모리가 충분하지 않으면 커널이 시작되지 않습니다.

The total number of warps Wblock in a block is as follows: )1,( size block W T ceilW   T is the number of threads per block,  Wsize is the warp size, which is equal to 32,  ceil(x, y) is equal to x rounded up to the nearest multiple of y. 
> 블록의 Wblock 워프 총 수는 다음과 같습니다: 1, (크기 블록 WT ceilW  T는 블록 당 스레드의 수이고, Wsize는 워프 크기이며 32와 동일하고, ceil(x, y)는 x의 가장 가까운 배수로 반올림한 x와 같습니다.

The total number of registers Rblock allocated for a block is as follows: 
> 블록에 할당된 레지스터 Rblock의 총 수는 다음과 같습니다.

GW is the warp allocation granularity, equal to 2 (compute capability 1.x only), Rk is the number of registers used by the kernel, GR is the register allocation granularity, which is equal to 256 for devices of compute capability 1.0 and 1.1, 512 for devices of compute capability 1.2 and 1.3, 64 for devices of compute capability 2.x, 256 for devices of compute capability 3.0. 
> GW는 2 (컴퓨팅 기능 1.x 만 해당)와 같은 워프 할당 세분성이고, Rk는 커널이 사용하는 레지스터의 수이고, GR은 컴퓨팅 기능 1.0 및 1.1의 디바이스에 대해 256과 같은 레지스터 할당 세분성입니다. 컴퓨팅 성능이 1.2와 1.3인 디바이스의 경우 512, 컴퓨팅 기능 2.x인 디바이스의 경우 64, 컴퓨팅 기능 3.0인 디바이스의 경우 256입니다.

The total amount of shared memory Sblock in bytes allocated for a block is as follows:
> 블록에 할당된 공유 메모리 Sblock의 총량은 바이트 단위로 다음과 같습니다.

Sk is the amount of shared memory used by the kernel in bytes, GS is the shared memory allocation granularity, which is equal to 512 for devices of compute capability 1.x, 128 for devices of compute capability 2.x, 256 for devices of compute capability 3.0. 
> Sk는 커널이 사용하는 공유 메모리의 양을 바이트 단위로 나타내고, GS는 컴퓨팅 기능 1.x의 디바이스에는 512이고, 컴퓨팅 기능 2.x의 디바이스에는 128개, 컴퓨팅 기능 3.0의 디바이스는 256개인 공유 메모리 할당 세분성입니다. 
