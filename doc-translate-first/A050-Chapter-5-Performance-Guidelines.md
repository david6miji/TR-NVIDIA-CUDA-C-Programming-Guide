## Chapter 5. Performance Guidelines 
> 5 장. 성능 지침

## 5.1 Overall Performance Optimization Strategies 
> 5.1 전반적인 성능 최적화 전략

Performance optimization revolves around three basic strategies:  
> 성능 최적화는 세 가지 기본 전략을 중심으로 이루어집니다.

Maximize parallel execution to achieve maximum utilization;  
> 최대 활용을 달성하기 위해 병렬 실행을 극대화하십시오.

Optimize memory usage to achieve maximum memory throughput;  
> 최대 메모리 처리량을 달성하기 위해 메모리 사용을 최적화하십시오.

Optimize instruction usage to achieve maximum instruction throughput. 
> 명령어 처리량을 최대화하기 위해 명령어 사용을 최적화하십시오.

Which strategies will yield the best performance gain for a particular portion of an application depends on the performance limiters for that portion; optimizing instruction usage of a kernel that is mostly limited by memory accesses will not yield any significant performance gain, for example. 
> 해당 부분의 성능 제한에 따른 애플리케이션의 특정 부분에 대해 어떤 전략은 최상의 성능 향상을 얻습니다. 메모리 액세스에 의해 대부분 제한되는 커널의 명령어 사용을 최적화해도  성능이 크게 향상되지는 않습니다.

Optimization efforts should therefore be constantly directed by measuring and monitoring the performance limiters, for example using the CUDA profiler. 
> 따라서 성능 제한기의 측정과 모니터링을 통해 최적화 작업을 지속적으로 수행해야 합니다 (예를 들어 CUDA 프로파일러를 사용하여).

Also, comparing the floating-point operation throughput or memory throughput – whichever makes more sense – of a particular kernel to the corresponding peak theoretical throughput of the device indicates how much room for improvement there is for the kernel.
> 또한 특정 커널의 부동 소수점 연산 처리량 또는 메모리 처리량을 비교하여 디바이스의 해당 피크의 이론적 처리량에 특정 커널의 부동 소수점 연산 처리량 또는 메모리 처리량을 비교하여 커널의 개선 여지가 어느 정도인지를 나타냅니다.

## 5.2 Maximize Utilization 
> 5.2 활용 극대화

To maximize utilization the application should be structured in a way that it exposes as much parallelism as possible and efficiently maps this parallelism to the various components of the system to keep them busy most of the time.  
> 활용도를 극대화하려면 가능한 한 병렬 처리가 가능한 방식으로 애플리케이션을 구성하고 이 병렬 처리를 시스템의 다양한 구성 요소에 효율적으로 매핑하여 대부분의 시간을 바쁘게 유지해야 합니다.

## 5.2.1 Application Level 
> 5.2.1 애플리케이션 레벨

At a high level, the application should maximize parallel execution between the host, the devices, and the bus connecting the host to the devices, by using asynchronous functions calls and streams as described in Section 3.2.5. 
> 최상위 레벨에서 애플리케이션은 3.2.5 절에서 설명한 비동기 함수 호출 및 스트림을 사용하여 호스트 및 디바이스와 호스트를 디바이스에 연결하는 버스 간의 병렬 실행을 최대화해야 합니다.

It should assign to each processor the type of work it does best: serial workloads to the host; parallel workloads to the devices. 
> 각 프로세서에 가장 잘 맞는 작업 유형을 할당해야 합니다. 호스트에 대한 일련의 작업 부하; 디바이스에 대한 병렬 작업 부하.

For the parallel workloads, at points in the algorithm where parallelism is broken because some threads need to synchronize in order to share data with each other, there are two cases: 
> 병렬 작업 부하의 경우, 일부 스레드가 서로 데이터를 공유하기 위해 동기화해야 하기 때문에 병렬 처리가 중단되는 알고리즘의 지점에는 두 가지 경우가 있습니다.

Either these threads belong to the same block, in which case they should use __syncthreads() and share data through shared memory within the same kernel invocation, or they belong to different blocks, in which case they must share data through global memory using two separate kernel invocations, one for writing to and one for reading from global memory. 
> 이 스레드는 동일한 블록에 속하며, 이 경우 __syncthreads()를 사용하여 동일한 커널 호출 내에서 공유 메모리를 통해 데이터를 공유하거나 서로 다른 블록에 속해야 합니다. 이 경우 두 개의 별도 커널 호출을 사용하여 전역 메모리를 통해 데이터를 공유해야 하는데, 하나는 전역 메모리에 쓰고 다른 하나는 전역 메모리에서 읽는 것입니다.

The second case is much less optimal since it adds the overhead of extra kernel invocations and global memory traffic. 
> 두 번째 경우는 여분의 커널 호출과 전역 메모리 트래픽의 오버헤드를 추가하기 때문에 훨씬 덜 최적입니다.

Its occurrence should therefore be minimized by mapping the algorithm to the CUDA programming model in such a way that the computations that require inter-thread communication are performed within a single thread block as much as possible. 
> 그러므로 스레드 간 통신을 필요로하는 계산이 가능한 한 단일 스레드 블록 내에서 수행되는 방식으로 알고리즘을 CUDA 프로그래밍 모델에 매핑하여 발생을 최소화해야 합니다.

## 5.2.2 Device Level 
> 5.2.2 디바이스 레벨

At a lower level, the application should maximize parallel execution between the multiprocessors of a device. 
> 하위 수준에서 애플리케이션은 디바이스의 다중프로세서 간의 병렬 실행을 최대화해야 합니다.

For devices of compute capability 1.x, only one kernel can execute on a device at one time, so the kernel should be launched with at least as many thread blocks as there are multiprocessors in the device. 
> 컴퓨팅 기능 1.x인 디바이스의 경우 한 번에 하나의 커널만 디바이스에서 실행할 수 있으므로 커널은 적어도 디바이스에 다중프로세서가 있는 만큼의 스레드 블록을 사용하여 시작해야 합니다.

For devices of compute capability 2.x and higher, multiple kernels can execute concurrently on a device, so maximum utilization can also be achieved by using streams to enable enough kernels to execute concurrently as described in Section 3.2.5. 
> 컴퓨팅 성능이 2.x 이상인 디바이스의 경우 디바이스에서 동시에 여러 커널을 실행할 수 있으므로 3.2.5 절에서 설명한 대로 커널을 동시에 실행할 수 있게 스트림을 사용하여 최대 활용도를 얻을 수 있습니다.

## 5.2.3 Multiprocessor Level 
> 5.2.3 다중프로세서 레벨

At an even lower level, the application should maximize parallel execution between the various functional units within a multiprocessor. 
> 더 낮은 레벨에서, 애플리케이션은 멀티프로세서 내의 다양한 기능 유닛 간의 병렬 실행을 최대화해야 합니다.

As described in Section 4.2, a GPU multiprocessor relies on thread-level parallelism to maximize utilization of its functional units. 
> 4.2 절에서 설명했듯이, GPU 멀티프로세서는 기능 단위의 활용을 극대화하기 위해 스레드 수준의 병렬 처리에 의존합니다.

Utilization is therefore directly linked to the number of resident warps. 
> 따라서 사용(이용)은 상주용 워프의 수와 직접적으로 관련됩니다.

At every instruction issue time, a warp scheduler selects a warp that is ready to execute its next instruction, if any, and issues the instruction to the active threads of the warp. 
> 모든 명령 발행 시간에, 워프 스케줄러는 다음 명령을 실행할 준비가 되어있는 워프를 선택하고, 워프의 활성 스레드에 그 명령을 발행합니다.

The number of clock cycles it takes for a warp to be ready to execute its next instruction is called the latency, and full utilization is achieved when all warp schedulers always have some instruction to issue for some warp at every clock cycle during that latency period, or in other words, when latency is completely “hidden”. 
> 워프가 다음 명령을 실행할 준비가 되기까지 소요되는 시간 사이클의 수를 대기 시간이라고하며, 모든 워프 스케줄러가 항상 대기 시간 동안, 또는 다른 말로 하면, 대기 시간이 완전히 "숨겨진" 때에, 매 시간 주기마다 일부 워프에 대해 어떤 명령을 내릴 때 전체 활용도가 달성됩니다. 

The number of instructions required to hide a latency of L clock cycles depends on the respective throughputs of these instructions (see Section 5.4.1 for the throughputs of various arithmetic instructions); assuming maximum throughput for all instructions, it is:  
> L 시간 사이클의 대기 시간을 숨기는 데 필요한 명령어의 수는 이러한 명령어의 각 처리량에 따라 다릅니다 (다양한 산술 명령어의 처리량은 5.4.1 절 참조). 모든 명령어에 대해 최대 처리량을 가정하면 다음과 같습니다.

L/4 (rounded up to nearest integer) for devices of compute capability 1.x since a multiprocessor issues one instruction per warp over four clock cycles, as mentioned in Section F.3.1,  
> F.3.1 절에서 언급한 것처럼, 다중프로세서가 4 시간 사이클에 걸쳐 워프 당 하나의 명령을 발행하기 때문에 컴퓨팅 기능 1.x인 디바이스에 대해 L/4 (가장 가까운 정수로 올림)

L for devices of compute capability 2.0 since a multiprocessor issues one instruction per warp over two clock cycles for two warps at a time, as mentioned in Section F.4.1,  
> 다중프로세서는 F.4.1 절에서 언급한 것처럼 한 번에 두 개의 워프에 대해 두 시간 사이클 동안 워프 당 하나의 명령어를 발행하기 때문에 컴퓨팅 기능 2.0인 디바이스에 대해 L이고,

2L for devices of compute capability 2.1 since a multiprocessor issues a pair of instructions per warp over two clock cycles for two warps at a time, as mentioned in Section F.4.1, 
> 다중프로세서는 F.4.1 절에서 언급한 것처럼 한 번에 두 개의 워프에 대해 두 시간 사이클 동안 워프 당 한 쌍의 명령어를 발행하기 때문에 컴퓨팅 기능 2.1인 디바이스에 대해 2L이고,

8L for devices of compute capability 3.0 since a multiprocessor issues a pair of instructions per warp over one clock cycle for four warps at a time, as mentioned in Section F.5.1. 
> 다중프로세서는 F.5.1 절에서 언급한 것처럼 한번에 4 개의 워프에 대해 하나의 시간 사이클 동안 워프 당 한 쌍의 명령어를 발행하기 때문에 컴퓨팅 기능 3.0인 디바이스에 대해서는 8L입니다.

For devices of compute capability 2.0, the two instructions issued every other cycle are for two different warps. 
> 컴퓨팅 기능 2.0인 디바이스의 경우 두 개의 다른 워프에 대해 매 사이클에 발행된 두 개의 명령어입니다.

For devices of compute capability 2.1, the four instructions issued every other cycle are two pairs for two different warps, each pair being for the same warp. 
> 컴퓨팅 기능 2.1인 디바이스의 경우, 두 개의 다른 워프에 매 사이클에 발행된 4개의 명령어는  2 쌍이며, 각 쌍은 동일한 워프에 속합니다.

For devices of compute capability 3.0, the eight instructions issued every cycle are four pairs for four different warps, each pair being for the same warp. 
> 컴퓨팅 기능 3.0인 디바이스의 경우, 두 개의 다른 워프에 대해 매 사이클마다 발행된 8 개의 명령어는 4 개의 쌍이며, 각 쌍은 동일한 워프에 속합니다.

The most common reason a warp is not ready to execute its next instruction is that the instruction’s input operands are not available yet. 
> 워프가 다음 명령어를 실행할 준비가 되지 않은 가장 보편적인 이유는 명령의 입력 피연산자가 아직 사용할 수 없기 때문입니다.

If all input operands are registers, latency is caused by register dependencies, i.e. some of the input operands are written by some previous instruction(s) whose execution has not completed yet. 
> 모든 입력 피연산자가 레지스터인 경우, 대기 시간은 레지스터 종속성에 의해 발생합니다. 즉, 일부 입력 피연산자는 실행이 아직 완료되지 않은 이전 명령어에 의해 작성됩니다.

In the case of a back-to-back register dependency (i.e. some input operand is written by the previous instruction), the latency is equal to the execution time of the previous instruction and the warp schedulers must schedule instructions for different warps during that time. 
> back-to-back 레지스터 종속성의 경우 (즉, 일부 입력 피연산자가 이전 명령에 의해 작성되는 경우) 대기 시간은 이전 명령의 실행 시간과 동일하며 워프 스케줄러는 그 시간 동안 다른 워프에 대한 명령을 예약해야 합니다.

Execution time varies depending on the instruction, but it is typically about 22 clock cycles for devices of compute capability 1.x and 2.x and about 11 clock cycles for devices of compute capability 3.0, which translates to 6 warps for devices of compute capability 1.x and 22 warps for devices of compute capability 2.x and higher (still assuming that warps execute instructions with maximum throughput, otherwise fewer warps are needed). 
> 실행 시간은 명령에 따라 다르지만 컴퓨팅 기능이 1.x 및 2.x인 디바이스의 경우 일반적으로 약 22 시간 사이클이고 컴퓨팅 기능 3.0인 디바이스의 경우 약 11 시간 사이클이며, 컴퓨팅 기능 1.x인 디바이스의 경우 6 워프로 바뀌며, 컴퓨팅 기능이 2.x 이상인 디바이스의 경우 22 워프로 바뀝니다 (여전히 최대 처리량의 명령어를 실행한다고 가정합니다. 그렇지 않으면 필요한 워프 수가 더 적습니다).

For devices of compute capability 2.1 and higher, this is also assuming enough instruction-level parallelism so that schedulers are always able to issue pairs of instructions for each warp. 
> 컴퓨팅 기능이 2.1 이상인 디바이스의 경우, 이는 또한 명령 레벨 병렬이 충분한 것으로 가정하므로 해당 스케줄러가 항상 각 워프에 대한 명령어 쌍을 발행할 수 있습니다.

If some input operand resides in off-chip memory, the latency is much higher: 400 to 800 clock cycles. 
> 일부 입력 피연산자가 오프-칩 메모리에 상주하는 경우 대기 시간은 400 ~ 800 시간 사이클보다 훨씬 높습니다.

The number of warps required to keep the warp schedulers busy during such high latency periods depends on the kernel code and its degree of instruction-level parallelism. 
> 이러한 높은 지연 시간 동안 워프 스케줄러를 바쁘게 유지하는 데 필요한 워프의 수는 커널 코드 및 명령 수준의 병렬 처리 수준에 따라 다릅니다.

In general, more warps are required if the ratio of the number of instructions with no off-chip memory operands (i.e. arithmetic instructions most of the time) to the number of instructions with off-chip memory operands is low (this ratio is commonly called the arithmetic intensity of the program). 
> 일반적으로 오프 칩 메모리 피연산자가 없는 명령어의 수 (즉, 대부분의 경우(시간)에 산술 명령어)와 오프 칩 메모리 피연산자가 있는 명령어의 수의 비율이 낮은 경우 (이 비율은 일반적으로 프로그램의 산술 강도라고 합니다) 더 많은 워프가 필요합니다.

If this ratio is 15, for example, then to hide latencies of about 600 clock cycles, about 10 warps are required for devices of compute capability 1.x and about 40 for devices of compute capability 2.x and higher (with the same assumptions as in the previous paragraph). 
> 예를 들어 이 비율이 15인 경우, 약 600 시간 사이클의 대기 시간을 숨기려면 컴퓨팅 기능 1.x의 디바이스에는 약 10개의 워프가 필요하고 컴퓨팅 기능 2.x 이상의 디바이스에는 약 40 개가 필요합니다 (동일한 가정하에 이전 절에서처럼).

Another reason a warp is not ready to execute its next instruction is that it is waiting at some memory fence (Section B.5) or synchronization point (Section B.6). 
> 워프가 다음 명령을 실행할 준비가 되지 않은 또 다른 이유는 일부 메모리 펜스 (B.5 절) 또는 동기화 포인트 (B.6 절)에서 기다리고 있다는 것입니다.

A synchronization point can force the multiprocessor to idle as more and more warps wait for other warps in the same block to complete execution of instructions prior to the synchronization point. 
> 동기화 지점은 동기화 지점 이전의 명령 실행을 완료하기 위해 같은 블록에서 점점 더 많은 워프가 다른 워프를 기다리면서 다중프로세서를 유휴 상태로 만들 수 있습니다.

Having multiple resident blocks per multiprocessor can help reduce idling in this case, as warps from different blocks do not need to wait for each other at synchronization points. 
> 다중프로세서 당 여러 개의 상주 블록을 사용하면 다른 블록의 워프가 동기화 지점에서 서로 기다릴 필요가 없으므로이 경우 유휴 시간을 줄일 수 있습니다.

The number of blocks and warps residing on each multiprocessor for a given kernel call depends on the execution configuration of the call (Section B.18), the memory resources of the multiprocessor, and the resource requirements of the kernel as described in Section 4.2. 
> 주어진 커널 호출에 대한 각 다중프로세서에 상주하는 블록 및 워프의 수는 4.2 절에서 설명한 대로 호출의 실행 구성 (B.18 절), 다중프로세서의 메모리 리소스 및 커널의 리소스 요구 사항에 따라 다릅니다.

To assist programmers in choosing thread block size based on register and shared memory requirements, the CUDA Software Development Kit provides a spreadsheet, called the CUDA Occupancy Calculator, where occupancy is defined as the ratio of the number of resident warps to the maximum number of resident warps (given in Appendix F for various compute capabilities). 
> 프로그래머가 레지스터 및 공유 메모리 요구 사항을 기반으로 스레드 블록 크기를 선택하는 것을 돕기 위해 CUDA 소프트웨어 개발 키트는 CUDA 점유율 계산기 (CUDA Occupancy Calculator)라는 스프레드 시트를 제공합니다. 이 점유율은 상주 워프의 최대 수에 대한 상주 워프 수의 비율로 정의됩니다 (다양한 컴퓨팅 기능에 대해서는 부록 F에 있습니다).

Register, local, shared, and constant memory usages are reported by the compiler when compiling with the --ptxas-options=-v option.  
> --ptxas-options=-v 옵션을 사용하여 컴파일할 때 레지스터, 로컬, 공유 및 상수 메모리 사용은 컴파일러에 의해 보고됩니다.

The total amount of shared memory required for a block is equal to the sum of the amount of statically allocated shared memory, the amount of dynamically allocated shared memory, and for devices of compute capability 1.x, the amount of shared memory used to pass the kernel’s arguments (see Section B.1.4). 
> 블록에 필요한 공유 메모리의 총량은 정적으로 할당된 공유 메모리의 양, 동적으로 할당된 공유 메모리의 양 및 컴퓨팅 기능 1.x인 디바이스의 경우, 커널 변수(B.1.4 절 참조)를 전달하는 데 사용된 공유 메모리의 양의 합과 같습니다. 

The number of registers used by a kernel can have a significant impact on the number of resident warps. 
> 커널이 사용하는 레지스터의 수는 상주하는 워프의 수에 큰 영향을 줄 수 있습니다.

For example, for devices of compute capability 1.2, if a kernel uses 16 registers and each block has 512 threads and requires very little shared memory, then two blocks (i.e. 32 warps) can reside on the multiprocessor since they require 2x512x16 registers, which exactly matches the number of registers available on the multiprocessor. 
> 예를 들어 컴퓨팅 기능이 1.2인 디바이스의 경우, 커널이 16 개의 레지스터를 사용하고 각 블록에 512 개의 스레드가 있고 공유 메모리가 거의 필요하지 않은 경우, 2x512x16 레지스터가 필요하기 때문에 두 개의 블록 (즉 32 개의 워프)이 다중프로세서에 있을 수 있습니다. 다중프로세서에서 사용 가능한 레지스터의 수와 정확히 일치합니다.

But as soon as the kernel uses one more register, only one block (i.e. 16 warps) can be resident since two blocks would require 2x512x17 registers, which are more registers than are available on the multiprocessor. 
> 그러나 커널이 하나 이상의 레지스터를 사용하자마자, 두 개의 블록은 다중프로세서에서 사용 가능한 것보다 많은 레지스터인 2x512x17 레지스터를 필요로하기 때문에 하나의 블록 (즉, 16 개의 워프)만 존재할 수 있습니다.

Therefore, the compiler attempts to minimize register usage while keeping register spilling (see Section 5.3.2.2) and the number of instructions to a minimum. 
> 따라서 컴파일러는 레지스터 유출 (5.3.2.2 절 참조) 및 명령어 수를 최소로 유지하면서 레지스터 사용을 최소화하려고 합니다.

Register usage can be controlled using the -maxrregcount compiler option or launch bounds as described in Section B.19. 
> 레지스터 사용은 -maxrregcount 컴파일러 옵션을 사용하거나 B.19 절에서 설명한 대로 론칭 범위를 사용하여 제어될 수 있습니다.

Each double variable (on devices that supports native double precision, i.e. devices of compute capability 1.2 and higher) and each long long variable uses two registers. 
> 각 이중 변수 (네이티브 배정밀도를 지원하는 디바이스에서, 즉 컴퓨팅 기능이 1.2 이상인 디바이스) 및 각 long long 변수는 두 개의 레지스터를 사용합니다.

However, devices of compute capability 1.2 and higher have at least twice as many registers per multiprocessor as devices with lower compute capability. 
> 그러나 컴퓨팅 기능이 1.2 이상인 디바이스는 컴퓨팅 기능이 낮은 디바이스처럼 다중프로세서 당 적어도 두 배의 레지스터가 있습니다.

The effect of execution configuration on performance for a given kernel call generally depends on the kernel code. Experimentation is therefore recommended. 
> 주어진 커널 호출의 성능에 대한 실행 구성의 영향은 일반적으로 커널 코드에 따라 다릅니다. 따라서 실험을 권장합니다.

Applications can also parameterize execution configurations based on register file size and shared memory size, which depends on the compute capability of the device, as well as on the number of multiprocessors and memory bandwidth of the device, all of which can be queried using the runtime (see reference manual). 
> 또한 애플리케이션은 레지스터 파일 크기와 공유 메모리 크기에 따라 실행 구성을 매개변수화할 수 있습니다. 이 크기는 디바이스의 다중프로세서 및 메모리 대역폭의 수 뿐만 아니라 디바이스의 컴퓨팅 기능에 따라 다릅니다. 이 모두는 런타임을 사용하여 쿼리할 수 있습니다 (참조 설명서 확인 ).

The number of threads per block should be chosen as a multiple of the warp size to avoid wasting computing resources with under-populated warps as much as possible. 
> 블록 당 스레드 수는 가능한 한 많이 부족한 워프로 컴퓨팅 리소스를 낭비하지 않도록 워프 크기의 배수로 선택해야 합니다.

## 5.3 Maximize Memory Throughput 
> 5.3 메모리 처리량 극대화

The first step in maximizing overall memory throughput for the application is to minimize data transfers with low bandwidth. 
> 애플리케이션의 전체 메모리 처리량을 최대화하는 첫 번째 단계는 낮은 대역폭으로 데이터 전송을 최소화하는 것입니다.

That means minimizing data transfers between the host and the device, as detailed in Section 5.3.1, since these have much lower bandwidth than data transfers between global memory and the device. 
> 이는 전역 메모리와 디바이스 간의 데이터 전송보다 훨씬 낮은 대역폭을 가지기 때문에, 5.3.1 절에서 설명한 것처럼 호스트와 디바이스 간의 데이터 전송을 최소화한다는 것을 의미합니다.

That also means minimizing data transfers between global memory and the device by maximizing use of on-chip memory: shared memory and caches (i.e. L1/L2 caches available on devices of compute capability 2.x and higher, texture cache and constant cache available on all devices). 
> 이는 또한 온-칩 메모리의 사용을 극대화함으로써 글로벌 메모리와 디바이스 간의 데이터 전송을 최소화한다는 것을 의미합니다: 공유 메모리 및 캐시 (즉, 컴퓨팅 기능이 2.x 이상인 디바이스에서 사용할 수있는 L1/L2 캐시, 텍스처 캐시 및 모든 디바이스에서 사용할 수 있는 상수 캐시).
 
Shared memory is equivalent to a user-managed cache: 
> 공유 메모리는 사용자 관리 캐시와 동일합니다.

The application explicitly allocates and accesses it. 
> 애플리케이션은 이를 명시적으로 할당하고 액세스합니다.

As illustrated in Section 3.2.3, a typical programming pattern is to stage data coming from device memory into shared memory; in other words, to have each thread of a block:  Load data from device memory to shared memory,  
> 3.2.3 절에서 설명한 것처럼 일반적인 프로그래밍 패턴은 디바이스 메모리에서 공유 메모리로 들어오는 데이터를 준비하는 것입니다. 즉, 블록의 각 스레드를 갖도록 : 디바이스 메모리에서 공유 메모리로 데이터를 로드합니다,

Synchronize with all the other threads of the block so that each thread can safely read shared memory locations that were populated by different threads,  
> 각 스레드가 다른 스레드로 채워진 공유 메모리 위치를 안전하게 읽을 수 있도록 블록의 다른 모든 스레드와 동기화하십시오.

Process the data in shared memory,  Synchronize again if necessary to make sure that shared memory has been updated with the results, Write the results back to device memory.
> 공유 메모리의 데이터를 처리하고, 필요한 경우 다시 동기화하여 공유 메모리가 결과로 업데이트되었는지 확인하고, 결과를 다시 디바이스 메모리에 작성합니다.

For some applications (e.g. for which global memory access patterns are data-dependent), a traditional hardware-managed cache is more appropriate to exploit data locality. 
> 일부 애플리케이션 (예를 들어, 전역 메모리 액세스 패턴이 데이터 종속적인 경우)의 경우, 전통적인 하드웨어 관리 캐시가 데이터 지역을 이용하는 것이 더 적합합니다.

As mentioned in Section F.4.1, for devices of compute capability 2.x and higher, the same on-chip memory is used for both L1 and shared memory, and how much of it is dedicated to L1 versus shared memory is configurable for each kernel call. 
> F.4.1 절에서 언급했듯이 컴퓨팅 기능이 2.x 이상인 디바이스의 경우, 동일한 온칩 메모리가 L1 및 공유 메모리에 사용되며 L1 대 공유 메모리에 할당되는 양은 각 커널 호출마다 구성 가능합니다.

The throughput of memory accesses by a kernel can vary by an order of magnitude depending on access pattern for each type of memory. 
> 커널에 의한 메모리 액세스의 처리량은 각 유형의 메모리에 대한 액세스 패턴에 따라 크기 순서에 따라 다를 수 있습니다.

The next step in maximizing memory throughput is therefore to organize memory accesses as optimally as possible based on the optimal memory access patterns described in Sections 5.3.2.1, 5.3.2.3, 5.3.2.4, and 5.3.2.5. 
> 따라서 메모리 처리량을 최대화하는 다음 단계는 5.3.2.1 절, 5.3.2.3 절, 5.3.2.4 절 및 5.3.2.5 절에서 설명한 최적의 메모리 액세스 패턴을 기반으로 가능한 한 최적으로 메모리 액세스를 구성하는 것입니다.

This optimization is especially important for global memory accesses as global memory bandwidth is low, so non-optimal global memory accesses have a higher impact on performance.  
> 이 최적화는 전역 메모리 대역폭이 낮아서 전역 메모리 액세스에 특히 중요하므로 비최적 전역 메모리 액세스는 성능에 더 큰 영향을 줍니다.

## 5.3.1 Data Transfer between Host and Device 
> 5.3.1 호스트와 디바이스 간의 데이터 전송

Applications should strive to minimize data transfer between the host and the device. 
> 애플리케이션은 호스트와 디바이스 간의 데이터 전송을 최소화하기 위해 노력해야 합니다.

One way to accomplish this is to move more code from the host to the device, even if that means running kernels with low parallelism computations. 
> 이것을 달성하는 한 가지 방법은 낮은 병렬 처리 계산으로 커널을 실행하는 경우에도 호스트에서 디바이스로 더 많은 코드를 이동하는 것입니다.

Intermediate data structures may be created in device memory, operated on by the device, and destroyed without ever being mapped by the host or copied to host memory. 
> 중간 데이터 구조는 디바이스 메모리에서 생성될 수 있고, 디바이스로 작동하며, 호스트에 의해 매핑되거나 호스트 메모리로 복사되지 않고 파손될 수 있습니다.

Also, because of the overhead associated with each transfer, batching many small transfers into a single large transfer always performs better than making each transfer separately. 
> 또한 각 전송과 관련된 오버헤드로 인해 다수의 작은 전송을 하나의 큰 전송으로 일괄 처리하는 것은 각 전송을 개별적으로 수행하는 것보다 항상 더 잘 수행됩니다.

On systems with a front-side bus, higher performance for data transfers between host and device is achieved by using page-locked host memory as described in Section 3.2.4. 
> 프런트 사이드 버스가 있는 시스템에서는 3.2.4 절에서 설명한 대로 페이지 잠금 호스트 메모리를 사용하여 호스트와 디바이스 간의 데이터 전송 성능을 향상시킵니다.

In addition, when using mapped page-locked memory (Section 3.2.4.3), there is no need to allocate any device memory and explicitly copy data between device and host memory. 
> 또한 매핑된 페이지 잠금 메모리 (3.2.4.3 절)를 사용할 때 디바이스 메모리를 할당하고 디바이스와 호스트 메모리 간에 데이터를 명시적으로 복사할 필요가 없습니다.

Data transfers are implicitly performed each time the kernel accesses the mapped memory. 
> 데이터 전송은 커널이 매핑된 메모리에 액세스할 때마다 암묵적으로 수행됩니다. 

For maximum performance, these memory accesses must be coalesced as with accesses to global memory (see Section 5.3.2.1). 
> 최대 성능을 위해서는 이러한 메모리 액세스가 전역 메모리에 대한 액세스와 같이 병합되어야 합니다 (섹션 5.3.2.1 참조).

Assuming that they are and that the mapped memory is read or written only once, using mapped page-locked memory instead of explicit copies between device and host memory can be a win for performance. 
> 매핑된 메모리가 한 번만 읽히거나 쓰여지고 있다고 가정하면 디바이스와 호스트 메모리 간에 명시적 복사 대신 매핑된 페이지 잠금 메모리를 사용하면 성능이 향상될 수 있습니다.

On integrated systems where device memory and host memory are physically the same, any copy between host and device memory is superfluous and mapped page-locked memory should be used instead. 
> 디바이스 메모리와 호스트 메모리가 물리적으로 동일한 통합 시스템에서는 호스트와 디바이스 메모리 간의 모든 복사가 불필요하며 매핑된 페이지 잠금 메모리가 대신 사용되어야 합니다.

Applications may query a device is integrated by checking that the integrated device property (see Section 3.2.6.1) is equal to 1. 
> 애플리케이션은 통합 디바이스 프로퍼티 (섹션 3.2.6.1 참조)가 1과 같은지 확인하여 디바이스가 통합되는지 쿼리할 수 있습니다.

## 5.3.2 Device Memory Accesses 
> 5.3.2 디바이스 메모리 액세스

An instruction that accesses addressable memory (i.e. global, local, shared, constant, or texture memory) might need to be re-issued multiple times depending on the distribution of the memory addresses across the threads within the warp. 
> 주소 지정 가능한 메모리 (즉, 전역, 지역, 공유, 상수 또는 텍스처 메모리)에 액세스하는 명령은 워프 내의 스레드에 걸친 메모리 어드레스의 분포에 따라 여러 번 재발행될 필요가 있을 수 있습니다.

How the distribution affects the instruction throughput this way is specific to each type of memory and described in the following sections. 
> 분산이 이러한 방식으로 명령 처리량에 미치는 영향은 각 메모리 유형에 따라 다르며 다음 절에서 설명합니다.

For example, for global memory, as a general rule, the more scattered the addresses are, the more reduced the throughput is. 
> 예를 들어, 전역 메모리의 경우 일반적으로 주소가 흩어질수록 처리량이 감소합니다.

## 5.3.2.1 Global Memory 
> 5.3.2.1 전역 메모리

Global memory resides in device memory and device memory is accessed via 32-, 64-, or 128-byte memory transactions. These memory transactions must be naturally aligned: 
> 전역 메모리는 디바이스 메모리에 상주하며 디바이스 메모리는 32, 64 또는 128 바이트 메모리 트랜잭션을 통해 액세스됩니다. 이러한 메모리 트랜잭션은 자연스럽게 정렬되어야 합니다.

Only the 32-, 64-, or 128-byte segments of device memory that are aligned to their size (i.e. whose first address is a multiple of their size) can be read or written by memory transactions. 
> 크기에 정렬된 (즉, 첫 번째 주소가 크기의 배수인) 디바이스 메모리의 32, 64 또는 128 바이트 세그먼트만 메모리 트랜잭션에 의해 읽거나 쓸 수 있습니다.

When a warp executes an instruction that accesses global memory, it coalesces the memory accesses of the threads within the warp into one or more of these memory transactions depending on the size of the word accessed by each thread and the distribution of the memory addresses across the threads. 
> 워프가 전역 메모리에 액세스하는 명령을 실행할 때, 각 스레드가 액세스하는 단어의 크기와 스레드 전체의 메모리 주소 분포에 따라 워프 내 스레드의 메모리 액세스를 이들 메모리 트랜잭션 중 하나 이상으로 병합합니다

In general, the more transactions are necessary, the more unused words are transferred in addition to the words accessed by the threads, reducing the instruction throughput accordingly. 
> 일반적으로, 더 많은 트랜잭션이 필요하며, 스레드들에 의해 액세스된 단어들에 더 많은 미사용 워드들이 전송되며, 이에 따라 명령 처리량이 감소합니다.

For example, if a 32-byte memory transaction is generated for each thread’s 4-byte access, throughput is divided by 8. 
> 예를 들어, 각 스레드의 4 바이트 액세스에 대해 32 바이트 메모리 트랜잭션이 생성되면 처리량은 8로 나뉩니다.

How many transactions are necessary and how much throughput is ultimately affected varies with the compute capability of the device. 
> 얼마나 많은 트랜잭션이 필요하며 얼마나 많은 처리량이 영향을 받는지는 궁극적으로 디바이스의 컴퓨팅 기능에 따라 다릅니다.

For devices of compute capability 1.0 and 1.1, the requirements on the distribution of the addresses across the threads to get any coalescing at all are very strict. 
> 컴퓨팅 기능이 1.0 및 1.1인 디바이스의 경우, 병합을 수행하기 위해 스레드에서 주소를 분배하는 요구 사항은 매우 엄격합니다.

They are much more relaxed for devices of higher compute capabilities. 
> 더 높은 컴퓨팅 기능을 갖춘 디바이스의 경우 훨씬 더 편안합니다.

For devices of compute capability 2.x and higher, the memory transactions are cached, so data locality is exploited to reduce impact on throughput. 
> 컴퓨팅 기능이 2.x 이상인 디바이스의 경우 메모리 트랜잭션이 캐시되므로 데이터 지역성을 활용하여 처리량에 대한 영향을 줄입니다.

Sections F.3.2, F.4.2, and F.5.2 give more details on how global memory accesses are handled for various compute capabilities. 
> F.3.2, F.4.2 및 F.5.2 절은 다양한 컴퓨팅 기능에 대해 전역 메모리 액세스를 처리하는 방법에 대해 자세히 설명합니다.

To maximize global memory throughput, it is therefore important to maximize coalescing by:  Following the most optimal access patterns based on Sections F.3.2 and F.4.2,  
> 따라서 전역 메모리 처리량을 최대화하려면 병합을 최대화하는 것이 중요합니다.
섹션 F.3.2 및 F.4.2를 기반으로 하는 최적의 액세스 패턴을 따릅니다.

Using data types that meet the size and alignment requirement detailed in Section 5.3.2.1.1,  Padding data in some cases, for example, when accessing a two-dimensional array as described in Section 5.3.2.1.2. 
> 5.3.2.1.1 절에 설명된 크기 및 정렬 요구 사항을 충족하는 데이터 유형을 사용하는 경우 (예를 들어 5.3.2.1.2 절에 설명된 것처럼 2 차원 배열에 액세스하는 경우) 경우에 따라 데이터를 패딩합니다. 

## 5.3.2.1.1 Size and Alignment Requirement 
> 5.3.2.1.1 크기 및 정렬 요구 사항

Global memory instructions support reading or writing words of size equal to 1, 2, 4, 8, or 16 bytes. 
> 전역 메모리 명령어는 1, 2, 4, 8 또는 16 바이트와 동일한 크기의 단어를 읽기 또는 쓰기를 지원합니다.

Any access (via a variable or a pointer) to data residing in global memory compiles to a single global memory instruction if and only if the size of the data type is 1, 2, 4, 8, or 16 bytes and the data is naturally aligned (i.e. its address is a multiple of that size). 
> 전역 메모리에 있는 데이터에 대한 (변수 또는 포인터를 통한) 모든 액세스는 데이터 유형의 크기가 1, 2, 4, 8 또는 16 바이트이고 데이터가 자연스럽게 정렬되는 경우에만 단일 전역 메모리 명령어로 컴파일됩니다 정렬 (즉, 주소는 해당 크기의 배수임).

If this size and alignment requirement is not fulfilled, the access compiles to multiple instructions with interleaved access patterns that prevent these instructions from fully coalescing. 
> 이 크기 및 정렬 요구 사항이 충족되지 않으면 액세스가 인터리브(상호 배치)된 액세스 패턴으로 여러 명령어로 컴파일되어 이러한 명령어가 완전히 통합되지 않도록 합니다.

It is therefore recommended to use types that meet this requirement for data that resides in global memory. 
> 따라서 전역 메모리에 있는 데이터에 이 요구 사항을 충족하는 형식을 사용하는 것이 좋습니다.

The alignment requirement is automatically fulfilled for the built-in types of Section B.3.1 like float2 or float4. 
> 정렬 요구 사항은 float2 또는 float4와 같은 B.3.1 절의 내장 유형에 대해 자동으로 충족됩니다.

For structures, the size and alignment requirements can be enforced by the compiler using the alignment specifiers __align__(8) or __align__(16), such as struct __align__(8) { float x;     float y; }; or struct __align__(16) { float x;  float y;  float z; }; 
> 구조체의 경우, 크기 및 정렬 요구 사항은 __align __ (8) {float x; float y; }; 또는 구조체 __align __ (16) {float x; float y; float z; }; 구조체와 같은 __align__(8) 또는  __align__(16) 인 정렬 지정자를 사용하여 컴파일러에 적용할 수 있습니다.

Any address of a variable residing in global memory or returned by one of the memory allocation routines from the driver or runtime API is always aligned to at least 256 bytes. 
> 전역 메모리에 있거나 드라이버 또는 런타임 API의 메모리 할당 루틴 중 하나가 반환하는 변수의 주소는 항상 최소 256 바이트로 정렬됩니다.

Reading non-naturally aligned 8-byte or 16-byte words produces incorrect results (off by a few words), so special care must be taken to maintain alignment of the starting address of any value or array of values of these types. 
> 자연스럽게 정렬되지 않은 8 바이트 또는 16 바이트 단어를 읽는 것은 잘못된 결과를 가져옵니다 (몇 마디로 꺼짐). 따라서 이러한 유형의 값 또는 값 배열의 시작 주소를 정렬하는 데 특별한 주의를 기울여야 합니다.

A typical case where this might be easily overlooked is when using some custom global memory allocation scheme, whereby the allocations of multiple arrays (with multiple calls to cudaMalloc() or cuMemAlloc()) is replaced by the allocation of a single large block of memory partitioned into multiple arrays, in which case the starting address of each array is offset from the block’s starting address. 
> 이것이 쉽게 간과될 수 있는 일반적인 경우는 커스텀 글로벌 메모리 할당 스키마를 사용할 때이고, 여러 개의 배열 (cudaMalloc() 또는 cuMemAlloc()에 대한 여러 호출로 할당)로 인하여 여러 개의 배열로 분할된 하나의 큰 메모리 블록 할당으로 대체됩니다. 이 경우 각 배열의 시작 주소는 블록의 시작 주소에서 오프셋됩니다.

## 5.3.2.1.2 Two-Dimensional Arrays 
> 5.3.2.1.2  2 차원 배열

A common global memory access pattern is when each thread of index (tx,ty) uses the following address to access one element of a 2D array of width width, located at address BaseAddress of type type* (where type meets the requirement described in Section 5.3.2.1.1):     
> 공통 전역 메모리 액세스 패턴은 인덱스 (tx, ty)의 각 스레드가 타입의 주소 BaseAddress에 있는 폭 너비의 2D 배열의 한 요소에 액세스 하려고, 다음 주소를 사용할 때 발생합니다
(타입이 5.3.2.1.1 절에 설명된 요구 사항을 충족시키는 경우).

For these accesses to be fully coalesced, both the width of the thread block and the width of the array must be a multiple of the warp size (or only half the warp size for devices of compute capability 1.x). 
> 이러한 액세스를 완전히 통합하려면 스레드 블록의 너비와 배열의 너비가 모두 워프 크기의 배수여야 합니다 (또는 컴퓨팅 기능 1.x인 디바이스의 경우 워프 크기의 절반이면 됩니다). 

In particular, this means that an array whose width is not a multiple of this size will be accessed much more efficiently if it is actually allocated with a width rounded up to the closest multiple of this size and its rows padded accordingly. 
> 특히 이것은 폭이 이 크기의 배수가 아닌 배열이 실제로 이 크기의 가장 가까운 배수로 반올림한 너비와 그에 따라 패딩된 행으로 할당된 경우 훨씬 효율적으로 액세스된다는 것을 의미합니다.

The cudaMallocPitch() and cuMemAllocPitch() functions and associated memory copy functions described in the reference manual enable programmers to write non-hardware-dependent code to allocate arrays that conform to these constraints. 
> 참조 매뉴얼에 설명된 cudaMallocPitch() 및 cuMemAllocPitch() 함수와 관련 메모리 복사 함수를 사용하면 프로그래머는 이러한 제약 사항을 준수하는 배열을 할당하기 위해 
비하드웨어 종속 코드를 쓸 수 있습니다.

## 5.3.2.2 Local Memory 
> 5.3.2.2 로컬 메모리

Local memory accesses only occur for some automatic variables as mentioned in Section B.2. 
> 로컬 메모리 액세스는 B.2 절에서 언급한 것처럼 일부 자동 변수에 대해서만 발생합니다.

Automatic variables that the compiler is likely to place in local memory are:  
> 컴파일러가 로컬 메모리에 배치할 가능성이 있는 자동 변수는 다음과 같습니다.

Arrays for which it cannot determine that they are indexed with constant quantities,  
> 일정한 수량으로 색인된 것으로 판별할 수 없는 배열,

Large structures or arrays that would consume too much register space,  
> 너무 많은 레지스터 공간을 소비하는 대형 구조 또는 배열,

Any variable if the kernel uses more registers than available (this is also known as register spilling). 
> 커널이 사용할 수 있는 많은 레지스터를 사용하는 경우의 변수입니다 (레지스터 누수라고도 함).

Inspection of the PTX assembly code (obtained by compiling with the –ptx or -keep option) will tell if a variable has been placed in local memory during the first compilation phases as it will be declared using the .local mnemonic and accessed using the ld.local and st.local mnemonics. 
> -ptx 또는 -keep 옵션을 사용하여 컴파일하여 얻은 PTX 어셈블리 코드를 검사하면 .local mnemonic을 사용하여 선언되고 id.local 및 st.local 니모닉을 사용하여 액세스되므로 변수가 첫 번째 컴파일 단계에서 로컬 메모리에 배치되었는지 알 수 있습니다.

Even if it has not, subsequent compilation phases might still decide otherwise though if they find it consumes too much register space for the targeted architecture: 
> 그렇지 않은 경우에도 후속 컴파일 단계에서는 대상 아키텍처에 너무 많은 레지스터 공간을 사용한다고 판단하더라도 다른 방법으로 결정할 수 있습니다. 

Inspection of the cubin object using cuobjdump will tell if this is the case.
> cuobjdump를 사용하여 cubin 객체를 검사하면 이것이 그 경우인지 알 수 있습니다.

Also, the compiler reports total local memory usage per kernel (lmem) when compiling with the --ptxas-options=-v option. 
> 또한 컴파일러는 --ptxas-options =-v 옵션을 사용하여 컴파일할 때 커널(lmem) 당 총 로컬 메모리 사용량을 보고합니다.

Note that some mathematical functions have implementation paths that might access local memory.  
> 일부 수학 함수에는 로컬 메모리에 액세스할 수 있는 구현 경로가 있습니다.

The local memory space resides in device memory, so local memory accesses have same high latency and low bandwidth as global memory accesses and are subject to the same requirements for memory coalescing as described in Section 5.3.2.1. 
> 로컬 메모리 공간은 디바이스 메모리에 상주하므로 로컬 메모리 액세스는 전역 메모리 액세스와 동일한 높은 대기 시간과 낮은 대역폭을 가지며 5.3.2.1 절에 설명된 것과 같이 메모리 병합에 대한 요구 사항이 동일합니다.

Local memory is however organized such that consecutive 32-bit words are accessed by consecutive thread IDs. 
> 그러나 연속 32 비트 단어가 연속된 스레드 ID에 의해 액세스되도록 로컬 메모리가 구성됩니다.

Accesses are therefore fully coalesced as long as all threads in a warp access the same relative address (e.g. same index in an array variable, same member in a structure variable). 
> 따라서 워프의 모든 스레드가 동일한 상대 주소 (예 : 배열 변수의 동일한 색인, 구조 변수의 동일한 구성원)에 액세스하는 한 액세스가 완전히 통합됩니다.

On devices of compute capability 2.x and higher, local memory accesses are always cached in L1 and L2 in the same way as global memory accesses (see Section F.4.2). 
> 컴퓨팅 기능이 2.x 이상인 디바이스에서 로컬 메모리 액세스는 항상 전역 메모리 액세스와 동일한 방식으로 L1 및 L2에서 캐시됩니다 (섹션 F.4.2 참조).

## 5.3.2.3 Shared Memory 
> 5.3.2.3 공유 메모리

Because it is on-chip, shared memory has much higher bandwidth and much lower latency than local or global memory. 
> 온칩이기 때문에 공유 메모리는 로컬 또는 전역 메모리보다 훨씬 더 높은 대역폭과 훨씬 낮은 대기 시간을 제공합니다.

To achieve high bandwidth, shared memory is divided into equally-sized memory modules, called banks, which can be accessed simultaneously. 
> 높은 대역폭을 달성하기 위해 공유 메모리는 뱅크라고 하는 동일한 크기의 메모리 모듈로 나누어져 있으며 동시에 액세스할 수 있습니다.

Any memory read or write request made of n addresses that fall in n distinct memory banks can therefore be serviced simultaneously, yielding an overall bandwidth that is n times as high as the bandwidth of a single module. 
> 따라서 n 개의 개별 메모리 뱅크에 속하는 n 개의 주소로 구성된 메모리 읽기 또는 쓰기 요청은 동시에 처리될 수 있으므로 단일 모듈의 대역폭만큼 높은 n 배인 전체 대역폭을 산출 할 수 있습니다.n 배인 전체 대역폭을 산출할 수 있습니다.

However, if two addresses of a memory request fall in the same memory bank, there is a bank conflict and the access has to be serialized. 
> 하지만 메모리 요청의 두 주소가 동일한 메모리 뱅크에 속하면 뱅크 충돌이 발생하며 액세스가 직렬화되어야 합니다.

The hardware splits a memory request with bank conflicts into as many separate conflict-free requests as necessary, decreasing throughput by a factor equal to the number of separate memory requests. 
> 하드웨어는 뱅크 충돌이 있는 메모리 요청을 필요한만큼 많은 별도의 충돌없는 요청으로 분할하여 별도의 메모리 요청 수와 동일한 비율로 처리량을 줄입니다. 

If the number of separate memory requests is n, the initial memory request is said to cause n-way bank conflicts. 
> 개별 메모리 요청의 수가 n인 경우 초기 메모리 요청은 n 방향 뱅크 충돌을 야기합니다.

To get maximum performance, it is therefore important to understand how memory addresses map to memory banks in order to schedule the memory requests so as to minimize bank conflicts. 
> 성능을 극대화하려면, 뱅크 충돌을 최소화하기 위해 메모리 요구를 예약하기 위해서 메모리 주소가 메모리 뱅크에 매핑되는 방법을 이해하는 것이 중요합니다.

This is described in Sections F.3.3, F.4.3, and F.5.3 for devices of compute capability 1.x, 2.x, and 3.0, respectively. 
> 이것은 컴퓨팅 기능 1.x, 2.x 및 3.0인 디바이스에 대해서는 F.3.3, F.4.3 및 F.5.3 절에 각각 설명되어 있습니다.

## 5.3.2.4 Constant Memory
> 5.3.2.4 상수 메모리
 
The constant memory space resides in device memory and is cached in the constant cache mentioned in Sections F.3.1 and F.4.1.  
> 상수 메모리 공간은 디바이스 메모리에 있으며 F.3.1 절 및 F.4.1 절에서 언급한 상수 캐시에 캐시됩니다.

For devices of compute capability 1.x, a constant memory request for a warp is first split into two requests, one for each half-warp, that are issued independently. 
> 컴퓨팅 기능 1.x인 디바이스의 경우, 워프에 대한 상수 메모리 요청은 먼저 독립적으로 발행되는 각각의 반 워프에 대한 두 개의 요청으로 분할됩니다.

A request is then split into as many separate requests as there are different memory addresses in the initial request, decreasing throughput by a factor equal to the number of separate requests. 
> 그런 다음 요청은 초기 요청에 다른 메모리 주소가 있는 만큼 많은 별도의 요청으로 분할되어 별도의 요청 수와 동일한 비율로 처리량이 감소합니다.

The resulting requests are then serviced at the throughput of the constant cache in case of a cache hit, or at the throughput of device memory otherwise. 
> 결과 요청은 캐시 적중의 경우 상수 캐시의 처리량에서 처리되고, 그렇지 않으면 디바이스 메모리의 처리량에서 처리됩니다.

## 5.3.2.5 Texture and Surface Memory 
> 5.3.2.5 텍스처 및 표면 메모리

The texture and surface memory spaces reside in device memory and are cached in texture cache, so a texture fetch or surface read costs one memory read from device memory only on a cache miss, otherwise it just costs one read from texture cache. 
> 텍스처 및 표면 메모리 공간은 디바이스 메모리에 상주하며 텍스처 캐시에 캐시되므로 
> 텍스처 페치 또는 표면 판독은 캐시 미스에서만 디바이스 메모리에서 읽은 하나의 메모리를 소비합니다. 그렇지 않으면 텍스처 캐시에서 하나의 읽기를 소비합니다.

The texture cache is optimized for 2D spatial locality, so threads of the same warp that read texture or surface addresses that are close together in 2D will achieve best performance. 
> 텍스처 캐시는 2D 공간적 지역성에 최적화되어 있으므로 2D에서 가까운 텍스처 또는 표면 주소를 읽는 동일한 워프의 스레드가 최상의 성능을 발휘합니다.

Also, it is designed for streaming fetches with a constant latency; a cache hit reduces DRAM bandwidth demand but not fetch latency. 
> 또한 일정한 대기 시간으로 스트리밍 가져오기를 위해 설계되었습니다. 캐시 히트는 DRAM 대역폭 요구를 줄이지만 대기 시간은 가져오지 않습니다.

Reading device memory through texture or surface fetching present some benefits that can make it an advantageous alternative to reading device memory from global or constant memory:  
> 텍스처 또는 표면 페칭(가져오기)을 통해 디바이스 메모리를 읽는 것은 전역 메모리 또는 상수 메모리에서 디바이스 메모리를 읽는 데 유리한 대안이 될 수 있는 몇 가지 이점을 제시합니다.

If the memory reads do not follow the access patterns that global or constant memory reads must respect to get good performance (see Sections 5.3.2.1 and 5.3.2.4), higher bandwidth can be achieved providing that there is locality in the texture fetches or surface reads (this is less likely for devices of compute capability 2.x and higher given that global memory reads are cached on these devices);  
> 메모리 읽기가 전역 또는 상수 메모리 읽기가 좋은 성능을 얻기 위해 중시해야 하는 액세스 패턴을 따르지 않으면 (5.3.2.1 절 및 5.3.2.4 절 참조) 텍스처 페치 또는 표면 읽기에 지역성이 있으면 높은 대역폭을 얻을 수 있습니다(이것은 전역 메모리 읽기가 이들 디바이스에서 캐시된 경우 컴퓨팅 기능이 2.x 이상인 디바이스에 대해서는 적습니다).

Addressing calculations are performed outside the kernel by dedicated units;  
> 어드레싱 계산은 전용 유닛에 의해 커널 외부에서 수행됩니다.

Packed data may be broadcast to separate variables in a single operation;  
> 가득찬 데이터는 단일 작업으로 변수를 분리하여 제공될 수 있습니다.

8-bit and 16-bit integer input data may be optionally converted to 32-bit floating-point values in the range [0.0, 1.0] or [-1.0, 1.0] (see Section 3.2.10.1.1). 
> 8 비트 및 16 비트 정수 입력 데이터는 [0.0, 1.0] 또는 [-1.0, 1.0] 범위의 32 비트 부동 소수점 값으로 선택적으로 변환될 수 있습니다 (3.2.10.1.1 절 참조).

## 5.4 Maximize Instruction Throughput 
> 5.4 명령어 처리량 극대화

To maximize instruction throughput the application should:  
> 명령 처리량을 극대화하려면 애플리케이션에서 다음을 수행해야 합니다.

Minimize the use of arithmetic instructions with low throughput; this includes trading precision for speed when it does not affect the end result, such as using intrinsic instead of regular functions (intrinsic functions are listed in Section C.2), single-precision instead of double-precision, or flushing denormalized numbers to zero;  
> 낮은 처리량으로 산술 명령어의 사용을 최소화하십시오. 이것은 일반 함수 (내장 함수는 C.2 절에 나열됩니다), 배 정밀도 대신 단 정밀도 또는 비정규화된 숫자를 0으로 플러시하는 대신 최종 결과에 영향을 미치지 않을 때 속도에 대한 거래 정밀도를 포함합니다 ;

Minimize divergent warps caused by control flow instructions as detailed in Section 5.4.2;  
> 5.4.2 절에서 상세히 설명된 것과 같이 제어 흐름 명령에 의해 야기된 확산적 워프를 최소화하십시오.

Reduce the number of instructions, for example, by optimizing out synchronization points whenever possible as described in Section 5.4.3 or by using restricted pointers as described in Section B.2.4. 
> 예를 들어 5.4.3 절에 설명된 대로 가능한 경우 동기화 지점을 최적화하거나 B.2.4 절에서 설명한 제한된 포인터를 사용하여 명령어 수를 줄이십시오. 

In this section, throughputs are given in number of operations per clock cycle per multiprocessor. 
> 이 절에서 처리량은 다중프로세서 당 시간 사이클 당 연산 수로 주어집니다.

For a warp size of 32, one instruction corresponds to 32 operations. 
> 워프 크기가 32인 경우 하나의 명령어가 32개의 연산에 해당합니다.

Therefore, if T is the number of operations per clock cycle, the instruction throughput is one instruction every 32/T clock cycles. All throughputs are for one multiprocessor. 
> 따라서, T가 시간 사이클 당 연산의 개수라면, 명령 처리량은 32/T 시간 사이클마다 하나의 명령이 됩니다. 모든 처리량은 하나의 다중프로세서를 위한 것입니다.

They must be multiplied by the number of multiprocessors in the device to get throughput for the whole device. 
> 전체 디바이스의 처리량을 얻으려면 디바이스의 다중프로세서 수를 곱해야 합니다.

## 5.4.1 Arithmetic Instructions 
> 5.4.1 산술 명령어

Table 5-1 gives the throughputs of the arithmetic instructions that are natively supported in hardware for devices of various compute capabilities. 
> 표 5-1은 다양한 컴퓨팅 기능을 갖춘 디바이스의 하드웨어에서 기본적으로 지원되는 산술 명령어의 처리량을 나타냅니다.

Table 5-1. Throughput of Native Arithmetic Instructions (Operations per Clock Cycle per Multiprocessor) 
> 표 5-1. 기본 산술 명령어의 처리량 (다중프로세서 당 시간 사이클 당 연산)
 
Throughput is lower for GeForce GPUs 
> GeForce GPU에 대한 처리량이 낮습니다

Other instructions and functions are implemented on top of the native instructions. 
> 기타 명령 및 기능은 기본 명령 위에 구현됩니다.

The implementation may be different for devices of different compute capabilities, and the number of native instructions after compilation may fluctuate with every compiler version. 
> 구현은 다른 컴퓨팅 기능을 가진 디바이스의 경우에는 다를 수 있으며 컴파일 후 기본 명령어의 수는 모든 컴파일러 버전에 따라 변동될 수 있습니다.

For complicated functions, there can be multiple code paths depending on input. 
> 복잡한 함수의 경우 입력에 따라 여러 개의 코드 경로가 있을 수 있습니다.

cuobjdump can be used to inspect a particular implementation in a cubin object. 
> cuobjdump는 cubin 객체의 특정 구현을 검사하는 데 사용될 수 있습니다.

The implementation of some functions are readily available on the CUDA header files (math_functions.h, device_functions.h, …). 
> 일부 함수의 구현은 CUDA 헤더 파일 (math_functions.h, device_functions.h, ...)에서 쉽게 사용할 수 있습니다.

In general, code compiled with -ftz=true (denormalized numbers are flushed to zero) tends to have higher performance than code compiled with -ftz=false. 
> 일반적으로 -ftz=true (비정규 숫자가 0으로 플러시 됨)로 컴파일된 코드는 -ftz=false로 컴파일된 코드보다 높은 성능을 나타내는 경향이 있습니다.

Similarly, code compiled with -prec-div=false (less precise division) tends to have higher performance code than code compiled with -prec-div=true, and code compiled with -prec-sqrt=false (less precise square root) tends to have higher performance than code compiled with -prec-sqrt=true. 
> 비슷하게 -prec-div=false로 컴파일된 코드 (정밀도가 떨어지는 경우)는 -prec-div=true로 컴파일된 코드보다 성능이 높은 경향이 있고 -prec-sqrt=false (덜 정확한 제곱근)로 컴파일된 코드는 -prec-sqrt=true로 컴파일된 코드보다 성능이 높은 경향이 있습니다.

The nvcc user manual describes these compilation flags in more details. 
> nvcc 사용자 설명서는 이러한 컴파일 플래그를 자세히 설명합니다.

Single-Precision Floating-Point Addition and Multiplication Intrinsics __fadd_r[d,u], __fmul_r[d,u], and __fmaf_r[n,z,d,u] (see Section C.2.1) compile to tens of instructions for devices of compute capability 1.x, but map to a single native instruction for devices of compute capability 2.x and higher. 
> 단 정밀도 부동 소수점 덧셈 및 곱셈 내장 함수 __fadd_r [d, u], __fmul_r [d, u] 및 __fmaf_r [n, z, d, u] (섹션 C.2.1 참조)는 컴퓨팅 기능 1.x인 디바이스에 대해 수십 개의 명령어로 컴파일하지만 컴퓨팅 기능 2.x 이상인 디바이스에 대해 단일 기본 명령어로 매핑합니다.

Single-Precision Floating-Point Division __fdividef(x, y) (see Section C.2.1) provides faster single-precision floating-point division than the division operator. 
> 단 정밀도 부동 소수점 분할 __fdividef(x, y) (C.2.1 절 참조)는 나누기 연산자보다 더 빠른 단 정밀도 부동 소수점 나누기를 제공합니다.

Single-Precision Floating-Point Reciprocal Square Root To preserve IEEE-754 semantics the compiler can optimize 1.0/sqrtf() into rsqrtf() only when both reciprocal and square root are approximate, (i.e. with -prec-div=false and -prec-sqrt=false). 
> 단 정밀도 부동 소수점 역 스퀘어 루트 IEEE-754 의미를 유지하기 위해 역수 및 제곱근이 근사치일 때만 (즉, -prec-div = false 및 -prec-sqrt = false를 사용하여) 컴파일러는 rsqrtf()에 1.0/sqrtf()를 최적화할 수 있습니다.

It is therefore recommended to invoke rsqrtf() directly where desired. 
> 따라서 원하는 위치에서 직접 rsqrtf()를 호출하는 것이 좋습니다.

Single-Precision Floating-Point Square Root 
> 단 정밀도 부동 소수점 제곱근

Single-precision floating-point square root is implemented as a reciprocal square root followed by a reciprocal instead of a reciprocal square root followed by a multiplication so that it gives correct results for 0 and infinity.  
> 단 정밀도 부동 소수점 제곱근은 역수 제곱근으로 구현되고 역수 제곱근 대신 역수가 따라 와서 곱셈이 수행되므로 0과 무한대에 대해 올바른 결과가 제공됩니다.

Sine and Cosine sinf(x), cosf(x), tanf(x), sincosf(x), and corresponding double-precision instructions are much more expensive and even more so if the argument x is large in magnitude. 
> sine과 Cosine sinf(x), cosf(x), tanf(x), sincosf(x) 및 해당 배 정밀도 명령어는 변수 x의 크기가 클 경우 해당하는 배 정밀도 명령어는 훨씬 더 비싸고 더욱 비쌉니다.

More precisely, the argument reduction code (see math_functions.h for implementation) comprises two code paths referred to as the fast path and the slow path, respectively. 
> 보다 정확하게는, 변수 축소 코드 (구현을 위해 math_functions.h 참조)는 각각 고속 경로 및 저속 경로라고 하는 두 개의 코드 경로를 포함합니다.

The fast path is used for arguments sufficiently small in magnitude and essentially consists of a few multiply-add operations. 
> 고속 경로는 크기가 충분히 작은 변수에 사용되며 본질적으로 몇 번의 곱셈-덧셈 연산으로 구성됩니다.

The slow path is used for arguments large in magnitude and consists of lengthy computations required to achieve correct results over the entire argument range. 
> 느린 경로는 크기가 큰 변수에 사용되며 전체 변수 범위에서 올바른 결과를 얻기 위해 필요한 긴 계산으로 구성됩니다.

At present, the argument reduction code for the trigonometric functions selects the fast path for arguments whose magnitude is less than 48039.0f for the single-precision functions, and less than 2147483648.0 for the double-precision functions. 
> 현재 삼각 함수에 대한 변수 감소 코드는 단 정밀도 함수의 경우 48039.0f보다 작은 크기의 변수의 경우 빠른 경로를 선택하고 배 정밀도 함수의 경우 2147483648.0보다 작은 변수의 빠른 경로를 선택합니다.

As the slow path requires more registers than the fast path, an attempt has been made to reduce register pressure in the slow path by storing some intermediate variables in local memory, which may affect performance because of local memory high latency and bandwidth (see Section 5.3.2.2). 
> 느린 경로는 빠른 경로보다 많은 레지스터가 필요하기 때문에, 로컬 메모리의 대기 시간 및 대역폭이 높기 때문에 성능에 영향을 줄 수 있으므로 (5.3.2.2 절 참조), 일부 중간 변수를 로컬 메모리에 저장하여 느린 경로의 레지스터 압력을 줄이려는 시도가 있었습니다

At present, 28 bytes of local memory are used by single-precision functions, and 44 bytes are used by double-precision functions. 
> 현재 단 정밀도 함수는 28 바이트의 로컬 메모리를 사용하고 배 정밀도 함수는 44 바이트를 사용합니다.

However, the exact amount is subject to change. 
> 그러나 정확한 양은 변경될 수 있습니다.

Due to the lengthy computations and use of local memory in the slow path, the throughput of these trigonometric functions is lower by one order of magnitude when the slow path reduction is required as opposed to the fast path reduction. 
> 긴 계산과 느린 경로에서의 로컬 메모리 사용으로 인해 빠른 경로 감소에 비해 느린 경로 감소가 필요한 경우, 이러한 삼각 함수의 처리량은 크기가 한 자리 수 더 낮습니다.

Integer Arithmetic 
> 정수 연산 

On devices of compute capability 1.x, 32-bit integer multiplication is implemented using multiple instructions as it is not natively supported. 
> 컴퓨팅 기능 1.x인 디바이스에서 32 비트 정수 곱셈은 기본적으로 지원되지 않으므로 여러 명령을 사용하여 구현됩니다.

24-bit integer multiplication is natively supported however via the __[u]mul24 intrinsic. 
> 24 비트 정수 곱셈은 기본적으로 __[u]mul24 내장 함수를 통해 지원됩니다.

Using __[u]mul24 instead of the 32-bit multiplication operator whenever possible usually improves performance for instruction bound kernels. 
> 가능할 때마다 32 비트 곱셈 연산자 대신 __[u]mul24를 사용하면 보통 명령 바운드 커널의 성능이 향상됩니다.

It can have the opposite effect however in cases where the use of __[u]mul24 inhibits compiler optimizations. 
> 그러나 __[u]mul24를 사용하면 컴파일러 최적화가 금지되는 경우에도 반대 효과가 발생할 수 있습니다.

On devices of compute capability 2.x and beyond, 32-bit integer multiplication is natively supported, but 24-bit integer multiplication is not. __[u]mul24 is therefore implemented using multiple instructions and should not be used. 
> 컴퓨팅 기능이 2.x 이상인 디바이스에서는 기본적으로 32 비트 정수 곱셈이 지원되지만 24 비트 정수 곱셈은 지원되지 않습니다. 따라서 __[u]mul24는 여러 명령어를 사용하여 구현되므로 사용하면 안됩니다.

Integer division and modulo operation are costly: tens of instructions on devices of compute capability 1.x, below 20 instructions on devices of compute capability 2.x and higher. 
> 정수 나누기 및 모듈 연산은 비용이 많이 듭니다. 컴퓨팅 기능 1.x인 디바이스에 수십 개의 명령어가 있고, 컴퓨팅 기능이 2.x 이상인 디바이스에 20 개 미만의 명령어가 있습니다.

They can be replaced with bitwise operations in some cases: If n is a power of 2, (i/n) is equivalent to (i>>log2(n)) and (i%n) is equivalent to (i&(n-1)); 
> n이 2의 거듭 제곱이면 (i/n)은 (i>>log2(n))과 같고 (i%n)은 (i&(n-1))이면, 경우에 따라 비트 연산으로 대체할 수 있습니다.

the compiler will perform these conversions if n is literal. 
> n이 리터럴인 경우 컴파일러에서 이러한 변환을 수행합니다.

__brev, __brevll, __popc, and __popcll compile to tens of instructions for devices of compute capability 1.x, but __brev and __popc map to a single instruction for devices of compute capability 2.x and higher __brev, __brevll, __popc, and __brevll and __popcll to just a few. 
> __brev, __brevll, __popc 및 __popcll은 컴퓨팅 기능 1.x의 디바이스에 대해 수십 개의 명령어로 컴파일되지만 __brev 및 __popc는 컴퓨팅 기능 2.x 이상인 디바이스에 대한 단일 명령어로 매핑되며 __brev, __brevll, __popc 및 __brevll와 __popcll은 몇 개만 매핑됩니다.

__clz, __clzll, __ffs, and __ffsll compile to fewer instructions for devices of compute capability 2.x and higher than for devices of compute capability 1.x. 
> __clz, __clzll, __ffs 및 __ffsll은 컴퓨팅 기능이 1.x 인 디바이스보다 컴퓨팅 기능이 2.x 이상인 디바이스에 대해 더 적은 수의 명령으로 컴파일됩니다. 

Type Conversion
> 유형 변환 

Sometimes, the compiler must insert conversion instructions, introducing additional execution cycles. This is the case for:  
> 가끔은 컴파일러에서 변환 명령을 삽입하여 추가 실행 주기를 도입해야 하는 경우가 있습니다. 다음과 같은 경우입니다.

Functions operating on variables of type char or short whose operands generally need to be converted to int,  Double-precision floating-point constants (i.e. those constants defined without any type suffix) used as input to single-precision floating-point computations (as mandated by C/C++ standards). 
> 피연산자가 일반적으로 int로 변환되어야 하는 char 또는 short 유형의 변수에서 작동하는 함수 단 정밀도 부동 소수점 계산(C/C++ 표준에서 위임한 대로)에 대한 입력으로 사용되는 배정 밀도 부동 소수점 상수 (즉, 유형 접미사 없이 정의된 상수) .

This last case can be avoided by using single-precision floating-point constants, defined with an f suffix such as 3.141592653589793f, 1.0f, 0.5f. 
> 이 마지막 경우는 3.141592653589793f, 1.0f, 0.5f와 같은 f 접미사로 정의된 단 정밀도 부동 소수점 상수를 사용하여 피할 수 있습니다.

## 5.4.2 Control Flow Instructions 
> 5.4.2 제어 흐름 명령

Any flow control instruction (if, switch, do, for, while) can significantly impact the effective instruction throughput by causing threads of the same warp to diverge (i.e. to follow different execution paths). 
> 임의의 흐름 제어 명령 (if, switch, do, for, while)은 동일한 워프의 스레드가 분기되도록 (즉, 다른 실행 경로를 따라가는) 효과적인 명령 처리량에 상당한 영향을 미칠 수있다.

If this happens, the different executions paths have to be serialized, increasing the total number of instructions executed for this warp. 
> 이 경우, 다른 실행 경로를 직렬화하여 이 워프를 위해 실행된 총 명령어 수를 늘려야 합니다.

When all the different execution paths have completed, the threads converge back to the same execution path. 
> 모든 다른 실행 경로가 완료되면 스레드는 다시 동일한 실행 경로로 수렴됩니다.

To obtain best performance in cases where the control flow depends on the thread ID, the controlling condition should be written so as to minimize the number of divergent warps. 
> 제어 흐름이 스레드 ID에 의존하는 경우 최상의 성능을 얻으려면, 다른 워프의 수가 최소화할 수 있도록 제어 조건을 작성해야 합니다.

This is possible because the distribution of the warps across the block is deterministic as mentioned in Section 4.1. 
> 이는 4.1 절에서 언급한 것처럼 블록을 통한 워프의 분배가 결정적이기 때문에 가능합니다.

A trivial example is when the controlling condition only depends on (threadIdx / warpSize) where warpSize is the warp size. 
> 간단한 예는 제어 조건이  warpSize가 워프 크기인 (threadIdx / warpSize)에만 의존할 때입니다.  

In this case, no warp diverges since the controlling condition is perfectly aligned with the warps. 
> 이 경우, 제어 상태가 워프와 완벽하게 정렬되기 때문에 워프는 나눠지지 않습니다.

Sometimes, the compiler may unroll loops or it may optimize out if or switch statements by using branch predication instead, as detailed below. 
> 가끔 컴파일러에서 루프를 풀거나 아래에 설명된 대로 분기 예측을 대신 사용하여 if 문 또는 switch 문을 최적화 할 수 있습니다.

In these cases, no warp can ever diverge. 
> 이러한 경우에는 워프가 나눠질 수 없습니다.

The programmer can also control loop unrolling using the #pragma unroll directive (see Section B.20). 
> 프로그래머는 #pragma unroll 지시문을 사용하여 루프 풀기를 제어할 수도 있습니다 (B.20 절 참조).

When using branch predication none of the instructions whose execution depends on the controlling condition gets skipped. 
> 분기 예측을 사용할 때 제어 조건에 따라 실행이 달라지는 명령어는 건너뛰지 않습니다.

Instead, each of them is associated with a per-thread condition code or predicate that is set to true or false based on the controlling condition and although each of these instructions gets scheduled for execution, only the instructions with a true predicate are actually executed. 
> 대신, 각각은 스레드 별 조건 코드나 제어 조건에 따라 참 또는 거짓으로 설정된 술어로 연관되며 이들 명령들 각각이 실행을 위해 스케줄링되지만, 참 술어가 있는 명령어만 실제로 실행됩니다.

Instructions with a false predicate do not write results, and also do not evaluate addresses or read operands.
> 거짓 술어가 있는 명령어는 결과를 쓰지 않으며 또한 주소 또는 피연산자를 읽지 않습니다.

The compiler replaces a branch instruction with predicated instructions only if the number of instructions controlled by the branch condition is less or equal to a certain threshold: If the compiler determines that the condition is likely to produce many divergent warps, this threshold is 7, otherwise it is 4. 
> 컴파일러는 분기 조건에 의해 제어되는 명령어의 수가 특정 임계 값보다 작거나 같은 경우에만 분기 명령어를 조건부 명령어로 대체합니다. 컴파일러에서 조건이 여러 가지 워프를 생성할 가능성이 있다고 판단하면 이 임계값은 7입니다. 아니면 4입니다.

## 5.4.3 Synchronization Instruction
> 5.4.3 동기화 명령
 
Throughput for __syncthreads() is 8 operations per clock cycle for devices of compute capability 1.x, 16 operations per clock cycle for devices of compute capability 2.x, and 128 operations per clock cycle for devices of compute capability 3.0. 
> __syncthreads()의 처리량은 컴퓨팅 기능 1.x인 디바이스의 경우 시간 사이클 당 8 개 연산이고 컴퓨팅 기능 2.x인 디바이스의 경우 시간 사이클 당 16 개 연산이며 컴퓨팅 기능 3.0의 디바이스인 경우 시간 사이클 당 128 개 연산입니다.

Note that __syncthreads() can impact performance by forcing the multiprocessor to idle as detailed in Section 5.2.3. 
> __syncthreads()는 5.2.3 절에 설명된 대로 다중프로세서를 유휴 상태로 강제 실행하여 성능에 영향을 줄 수 있습니다.

Because a warp executes one common instruction at a time, threads within a warp are implicitly synchronized and this can sometimes be used to omit __syncthreads() for better performance. 
> 워프는 한 번에 하나의 공통 명령을 실행하므로 워프 내의 스레드는 암시적으로 동기화되므로 더 나은 성능을 위해 __syncthreads()를 생략할 때 이 명령을 사용할 수 있습니다.

In the following code sample, for example, there is no need to call __syncthreads() after each of the additions performed within the body of the “if (tid < 32) { }“ statement since they operate within a single warp (this is assuming the size of a warp is 32). 
> 예를 들어 다음 코드 샘플에서는 "if (tid <32) {}"문의 본문 내에서 수행된 각 추가 작업 후에 단일 워프 내에서 작동하기 때문에 __syncthreads()를 호출할 필요가 없습니다 (이것은 워프의 크기가 32라고 가정할 때입니다).

Simply removing the __syncthreads() is not enough however; smem must also be declared as volatile as described in Section D.2.1.2. 
> 단순히 __syncthreads()를 제거하는 것만으로는 충분하지 않습니다. smem은 D.2.1.2 절에서 설명한 것처럼 불안정하게 선언되어야 합니다. 
