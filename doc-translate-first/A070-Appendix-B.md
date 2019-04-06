# Appendix B. C Language Extensions 
> 부록 B. C 언어 확장

## B.1 Function Type Qualifiers
> B.1 함수 유형 한정자
 
Function type qualifiers specify whether a function executes on the host or on the device and whether it is callable from the host or from the device. 
> 함수 유형 한정자는 함수가 호스트에서 또는 디바이스에서 실행되는지, 그리고 호스트에서 호출 가능한지 또는 디바이스에서 호출 가능한지 여부를 지정합니다.

## B.1.1 __device__ 
> B.1.1 __device__

The __device__ qualifier declares a function that is:  Executed on the device.  
Callable from the device only. 
> __device__ 한정자는 다음과 같은 함수를 선언합니다. 디바이스에서 실행합니다. 디바이스에서만 호출 가능합니다.

## B.1.2 __global__ 
> B.1.2 __global__

The __global__ qualifier declares a function as being a kernel. 
> __global__ 한정자는 함수를 커널로 선언합니다.

Such a function is:  Executed on the device,  Callable from the host only. 
__global__ functions must have void return type. 
> 이러한 기능은 다음과 같습니다. 디바이스에서 실행되며 호스트에서만 호출 가능합니다. __global__ 함수는 반환 형식이 void여야합니다.

Any call to a __global__ function must specify its execution configuration as described in Section B.18. 
> __global__ 함수에 대한 호출은 B.18 절에서 설명한 대로 실행 구성을 지정해야 합니다.

A call to a __global__ function is asynchronous, meaning it returns before the device has completed its execution. 
> __global__ 함수 호출은 비동기식입니다. 즉, 디바이스가 실행을 완료하기 전에 반환합니다.

## B.1.3 __host__ 
> B.1.3 __host__

The __host__ qualifier declares a function that is:  Executed on the host,  Callable from the host only. 
> __host__ 한정자는 다음과 같은 함수를 선언합니다. 호스트에서 실행되고 호스트에서만 호출 가능합니다.

It is equivalent to declare a function with only the __host__ qualifier or to declare it without any of the __host__, __device__, or __global__ qualifier; in either case the function is compiled for the host only. 
> __host__ 한정자만 있는 함수를 선언하거나 __host__, __device__ 또는 __global__ 한정자 없이 함수를 선언하는 것과 같습니다. 두 경우 모두 함수는 호스트에 대해서만 컴파일됩니다.

The __global__ and __host__ qualifiers cannot be used together. 
> __global__ 및 __host__ 한정자는 함께 사용할 수 없습니다.

The __device__ and __host__ qualifiers can be used together however, in which case the function is compiled for both the host and the device. 
> 그러나 __device__ 및 __host__ 한정자는 함께 사용할 수 있지만, 이 경우 함수는 호스트와 디바이스 모두에 대해 컴파일됩니다.

The __CUDA_ARCH__ macro introduced in Section 3.1.4 can be used to differentiate code paths between host and device: 
> 3.1.4 절에서 소개한 __CUDA_ARCH__ 매크로는 호스트와 디바이스 사이의 코드 경로를 구별하는 데 사용할 수 있습니다.

## B.1.4 __noinline__ and __forceinline__ 
> B.1.4 __noinline__ 및 __forceinline__

When compiling code for devices of compute capability 1.x, a __device__ function is always inlined by default. 
> 컴퓨팅 기능 1.x의 디바이스인 경우 코드를 컴파일할 때 __device__ 함수는 항상 기본적으로 인라인됩니다.

When compiling code for devices of compute capability 2.x and higher, a __device__ function is only inlined when deemed appropriate by the compiler. 
> 컴퓨팅 기능 2.x 이상의 디바이스인 코드를 컴파일할 때 __device__ 함수는 컴파일러가 적절하다고 판단할 때만 인라인됩니다.

The __noinline__ function qualifier can be used as a hint for the compiler not to inline the function if possible. 
> __noinline__ 함수 한정자는 가능한 경우 컴파일러가 함수를 인라인하지 않도록 하는 힌트로 사용할 수 있습니다.

The function body must still be in the same file where it is called. 
> 함수 본문은 호출된 동일한 파일에 있어야 합니다.

For devices of compute capability 1.x, the compiler will not honor the __noinline__ qualifier for functions with pointer parameters and for functions with large parameter lists. 
> 컴퓨팅 기능 1.x의 디바이스인 경우 컴파일러는 포인터 매개변수가 있는 함수와 큰 매개변수 목록이 있는 함수에 대해 __noinline__ 한정자를 중시하지 않습니다.

For devices of compute capability 2.x and higher, the compiler will always honor the __noinline__ qualifier. 
> 컴퓨팅 기능 2.x 이상의 디바이스인 경우 컴파일러는 항상 __noinline__ 한정자를 중시합니다.

The __forceinline__ function qualifier can be used to force the compiler to inline the function. 
> __forceinline__ 함수 한정자를 사용하여 컴파일러에서 함수를 인라인하게 만들 수 있습니다.

B.2 Variable Type Qualifiers 
> B.2 변수 유형 한정자

Variable type qualifiers specify the memory location on the device of a variable. 
> 변수 유형 한정자는 변수의 디바이스에서 메모리 위치를 지정합니다.

An automatic variable declared in device code without any of the __device__, __shared__ and __constant__ qualifiers described in this section generally resides in a register. 
> 이 절에서 설명하는 __device__, __shared__ 및 __constant__ 한정자 없이 디바이스 코드에서 선언된 자동 변수는 일반적으로 레지스터에 있습니다.

However in some cases the compiler might choose to place it in local memory, which can have adverse performance consequences as detailed in Section 5.3.2.2. 
> 그러나 어떤 경우 컴파일러는 이를 로컬 메모리에 배치하기로 결정할 수 있으며, 이것은 5.3.2.2 절에서 설명한 것처럼 성능에 불리한 영향을 줄 수 있습니다.
 
## B.2.1 __device__ 
> B.2.1 __device__

The __device__ qualifier declares a variable that resides on the device. 
> __device__ 한정자는 디바이스에 상주하는 변수를 선언합니다.

At most one of the other type qualifiers defined in the next three sections may be used together with __device__ to further specify which memory space the variable belongs to. 
> 다음 세 절에서 정의한 다른 유형 한정자 중 하나만 __device__와 함께 사용하여 변수가 속하는 메모리 공간을 더 자세히 지정할 수 있습니다.

If none of them is present, the variable:  Resides in global memory space,  
> 존재하지 않으면 변수는 다음과 같습니다. 전역 메모리 공간에 상주합니다.

Has the lifetime of an application,  Is accessible from all the threads within the grid and from the host through the runtime library (cudaGetSymbolAddress() / cudaGetSymbolSize() / cudaMemcpyToSymbol() / cudaMemcpyFromSymbol()). 
> 애플리케이션의 수명이 있으며, 그리드 내의 모든 스레드와 런타임 라이브러리 (cudaGetSymbolAddress() / cudaGetSymbolSize() / cudaMemcpyToSymbol() / cudaMemcpyFromSymbol())를 통해 호스트에서 액세스할 수 있습니다.

## B.2.2 __constant__ 
> B.2.2 __constant__

The __constant__ qualifier, optionally used together with __device__, declares a variable that:  Resides in constant memory space,  
> 선택적으로 __device__와 함께 사용되는 __constant__ 한정자는 다음과 같은 변수를 선언합니다. 상수 메모리 공간에 상주합니다,

Has the lifetime of an application,  
Is accessible from all the threads within the grid and from the host through the runtime library (cudaGetSymbolAddress() / cudaGetSymbolSize() / cudaMemcpyToSymbol() / cudaMemcpyFromSymbol()). 
> 애플리케이션의 수명이 있으며, 그리드 내의 모든 스레드와 런타임 라이브러리 (cudaGetSymbolAddress() / cudaGetSymbolSize() / cudaMemcpyToSymbol() / cudaMemcpyFromSymbol())를 통해 호스트에서 액세스할 수 있습니다.

## B.2.3 __shared__ 
> B.2.3 __shared__

The __shared__ qualifier, optionally used together with __device__, declares a variable that:  Resides in the shared memory space of a thread block,  
> 선택적으로 __device__와 함께 사용되는 __shared__ 한정자는 다음을 선언하는 변수를 선언합니다. 스레드 블록의 공유 메모리 공간에 상주합니다,

Has the lifetime of the block,  Is only accessible from all the threads within the block. 
> 블록의 수명은 블록 내의 모든 스레드에서만 액세스할 수 있습니다.

When declaring a variable in shared memory as an external array such as extern__shared__ float shared[]; the size of the array is determined at launch time (see Section B.18). 
> extern __shared__ float shared[]와 같은 외부 배열로 공유 메모리의 변수를 선언할 때; 배열의 크기는 론칭 시 결정됩니다 (B.18 절 참조).

All variables declared in this fashion, start at the same address in memory, so that the layout of the variables in the array must be explicitly managed through offsets. 
> 이런 식으로 선언된 모든 변수는 메모리의 동일한 주소에서 시작하므로 배열의 변수 레이아웃을 오프셋을 통해 명시적으로 관리해야 합니다.

For example, if one wants the equivalent of short array0[128]; float array1[64]; int   array2[256]; in dynamically allocated shared memory, one could declare and initialize the arrays the following way: 
> 예를 들어, short array0[128]; float array1[64]; int array2[256]; 동적으로 할당된 공유 메모리에서 다음과 같이 배열을 선언하고 초기화할 수 있습니다.

Note that pointers need to be aligned to the type they point to, so the following code, for example, does not work since array1 is not aligned to 4 bytes. 
> 포인터는 포인터가 가리키는 유형으로 정렬되어야 합니다. 예를 들어 array1이 4 바이트로 정렬되지 않았으므로 다음 코드가 작동하지 않습니다.

Alignment requirements for the built-in vector types are listed in Table B-1. 
> 내장 벡터 유형의 정렬 요구 사항은 표 B-1에 나열되어 있습니다.

B.2.4 __restrict__ 
> B.2.4 __restrict__

nvcc supports restricted pointers via the __restrict__ keyword. 
> nvcc는 __restrict__ 키워드를 통해 제한된 포인터를 지원합니다.

Restricted pointers were introduced in C99 to alleviate the aliasing problem that exists in C-type languages, and which inhibits all kind of optimization from code reordering to common sub-expression elimination. 
> 제한된 포인터는 C-타입 언어에 존재하는 앨리어싱 문제를 줄이기 위해 C99에서 도입되었으며 모든 종류의 최적화가 코드 재정렬에서 일반적인 하위 표현 제거로 억제됩니다.

Here is an example subject to the aliasing issue, where use of restricted pointer can help the compiler to reduce the number of instructions: 
> 다음은 앨리어싱 문제의 예입니다. 제한된 포인터를 사용하면 컴파일러가 명령어 수를 줄일 수 있습니다.

In C-type languages, the pointers a, b, and c may be aliased, so any write through c could modify elements of a or b. 
> C 유형 언어에서 포인터 a, b 및 c는 별명이 있을 수 있으므로 c를 통한 모든 쓰기는 a 또는 b의 요소를 수정할 수 있습니다.

This means that to guarantee functional correctness, the compiler cannot load a[0] and b[0] into registers, multiply them, and store the result to both c[0] and c[1], because the results would differ from the abstract execution model if, say, a[0] is really the same location as c[0]. 
> 즉, 기능상의 정확성을 보장하기 위해 컴파일러는 a[0]과 b[0]을 레지스터에 로드하고 곱한 다음 결과를 c[0]과 c[1]에 모두 저장할 수 없습니다. a[0]이 사실 c[0]과 같은 위치인 경우에는, 말하자면, 결과가 추상 실행 모델과 다르기 때문입니다. 

So the compiler cannot take advantage of the common sub-expression. 
> 따라서 컴파일러는 공통 하위 표현식을 이용할 수 없습니다.

Likewise, the compiler cannot just reorder the computation of c[4] into the proximity of the computation of c[0] and c[1] because the preceding write to c[3] could change the inputs to the computation of c[4]. 
> 마찬가지로, 앞의 c[3]에 대한 쓰기가 입력을 c[4]의 계산으로 변경할 수 있기 때문에, 컴파일러는 c[4]의 계산을 c[0] 및 c[1] 계산의 근접으로 재배열할 수 없습니다.

By making a, b, and c restricted pointers, the programmer asserts to the compiler that the pointers are in fact not aliased, which in this case means writes through c would never overwrite elements of a or b. 
> a, b 및 c로 제한된 포인터를 작성함으로써 프로그래머는 포인터가 실제로 별칭이 지정되지 않았음을 컴파일러에게 알립니다. 이 경우에는 c를 통한 쓰기는 a 또는 b의 요소를 절대 덮어쓰지 않습니다.

This changes the function prototype as follows: 
> 이것은 다음과 같이 함수 프로토타입을 변경합니다.
 
Note that all pointer arguments need to be made restricted for the compiler optimizer to derive any benefit. 
> 모든 포인터 변수는 컴파일러 옵티마이저가 모든 이점을 끌어내기 위해 제한되어야 합니다.

With the __restrict keywords added, the compiler can now reorder and do common sub-expression elimination at will, while retaining functionality identical with the abstract execution model: 
> __restrict 키워드가 추가되면 컴파일러는 추상적 실행 모델과 동일한 기능을 유지하면서 공통 하위 표현식 제거를 자유롭게 재정렬하고 수행할 수 있습니다.

The effects here are a reduced number of memory accesses and reduced number of computations. 
> 여기서의 효과는 메모리 액세스 횟수 감소 및 계산 횟수 감소입니다.

This is balanced by an increase in register pressure due to "cached" loads and common sub-expressions. 
> 이것은 "캐시된" 로드 및 공통 하위 표현식으로 인해 레지스터 압박이 증가하여 균형을 이룹니다.

Since register pressure is a critical issue in many CUDA codes, use of restricted pointers can have negative performance impact on CUDA code, due to reduced occupancy. 
> 레지스터 압박은 많은 CUDA 코드에서 중요한 문제이므로 제한된 포인터를 사용하면 점유가 줄어 CUDA 코드에 부정적인 영향을 미칠 수 있습니다.

## B.3 Built-in Vector Types 
> B.3 내장 벡터 유형

B.3.1 char1, uchar1, char2, uchar2, char3, uchar3, char4, uchar4, short1, ushort1, short2, ushort2, short3, ushort3, short4, ushort4, int1, uint1, int2, uint2, int3, uint3, int4, uint4, long1, ulong1, long2, ulong2, long3, ulong3, long4, ulong4, longlong1, ulonglong1, longlong2, ulonglong2, float1, float2, float3, float4, double1, double2 

These are vector types derived from the basic integer and floating-point types. 
> 이들은 기본 정수 및 부동 소수점 유형에서 파생된 벡터 유형입니다.

They are structures and the 1st, 2nd, 3rd, and 4th components are accessible through the fields x, y, z, and w, respectively. 
> 이것들은 구조체이고, 1, 2, 3, 4 번째 구성 요소는 각각 x, y, z 및 w 필드를 통해 액세스할 수 있습니다.

They all come with a constructor function of the form make_<type name>; for example, int2 make_int2(int x, int y); which creates a vector of type int2 with value (x, y). 
> 이것들은 모두 make_ <type name> 형식의 생성자 함수와 함께 제공됩니다. 예를 들어, int2 make_int2 (int x, int y); 값이 (x, y) 인 int2 유형의 벡터를 만듭니다.
 
In host code, the alignment requirement of a vector type is equal to the alignment requirement of its base type. 
> 호스트 코드에서 벡터 유형의 정렬 요구 사항은 기본 유형의 정렬 요구 사항과 같습니다.

This is not always the case in device code as detailed in Table B-1. 
> 표 B-1에 설명된 대로 디바이스 코드에서 항상 그런 것은 아닙니다.

Table B-1. Alignment Requirements in Device Code 
> 표 B-1. 디바이스 코드의 정렬 요구 사항

Type Alignment char1, uchar1 1 char2, uchar2 2 char3, uchar3 1 char4, uchar4 4 short1, ushort1 2 short2, ushort2 4 short3, ushort3 2 short4, ushort4 8 int1, uint1 4 int2, uint2 8 int3, uint3 4 int4, uint4 16 long1, ulong1 4 

if sizeof(long) is equal to sizeof(int), 8, otherwise long2, ulong2 8 if sizeof(long) is equal to sizeof(int), 16, otherwise long3, ulong3 4 if sizeof(long) is equal to sizeof(int), 8, otherwise long4, ulong4 16 longlong1, ulonglong1 8 longlong2, ulonglong2 16 float1 4 float2 8 float3 4 float4 16 double1 8 double2 16 
> sizeof(long)가 sizeof(int)와 같은 경우 8, 그렇지 않으면 long2, ulong2 sizeof(long)이 sizeof(int)와 같은 경우 16, 그렇지 않으면 long3, ulong3 sizeof (long)이 sizeof int), 8, 그렇지 않으면 long4, ulong4 16 longlong1, ulonglong1 8 longlong2, ulonglong2 16 float1 4 float2 8 float3 4 float4 16 double1 8 double2 16

## B.3.2 dim3 
> B.3.2 dim3

This type is an integer vector type based on uint3 that is used to specify dimensions. 
> 이 유형은 차원을 지정하는 데 사용되는 uint3을 기반으로 하는 정수 벡터 유형입니다. 

When defining a variable of type dim3, any component left unspecified is initialized to 1. 
> dim3 유형의 변수를 정의 할 때 지정되지 않은 모든 구성 요소는 1로 초기화됩니다.

## B.4 Built-in Variables
> B.4 내장 변수
 
Built-in variables specify the grid and block dimensions and the block and thread indices. 
> 내장(기본 제공) 변수는 그리드 및 블록 차원과 블록 및 스레드 인덱스를 지정합니다. 

They are only valid within functions that are executed on the device. 
> 이들은 디바이스에서 실행되는 함수 내에서만 유효합니다.
 
## B.4.1 gridDim 
> B.4.1 gridDim 

This variable is of type dim3 (see Section B.3.2) and contains the dimensions of the grid. 
> 이 변수는 dim3 유형이며 (B.3.2 절 참조) 그리드의 크기(차원)를 포함합니다.

## B.4.2 blockIdx
> B.4.2 blockIdx

This variable is of type uint3 (see Section B.3.1) and contains the block index within the grid. 
> 이 변수는 uint3 유형이며 (B.3.1 절 참조) 그리드 내의 블록 인덱스를 포함합니다.

## B.4.3 blockDim 
> B.4.3 blockDim 

This variable is of type dim3 (see Section B.3.2) and contains the dimensions of the block. 
> 이 변수는 dim3 유형이며 (B.3.2 절 참조) 블록의 크기를 포함합니다.

## B.4.4 threadIdx 
> B.4.4 threadIdx 

This variable is of type uint3 (see Section B.3.1) and contains the thread index within the block. 
> 이 변수는 uint3 유형이며 (B.3.1 절 참조) 블록 내의 스레드 인덱스를 포함합니다.

## B.4.5 warpSize 
> B.4.5 warpSize 

This variable is of type int and contains the warp size in threads (see Section 4.1 for the definition of a warp). 
> 이 변수는 int 유형이며 스레드의 워프 크기를 포함합니다 (워프 정의에 대해서는 4.1 절 참조).

## B.5 Memory Fence Functions 
> B.5 메모리 펜스 함수

void __threadfence_block(); waits until all global and shared memory accesses made by the calling thread prior to __threadfence_block() are visible to all threads in the thread block. 
> void __threadfence_block(); __threadfence_block() 이전의 호출 스레드가 만든 모든 전역 및 공유 메모리 액세스가 스레드 블록의 모든 스레드에서 볼 수 있을 때까지 대기합니다. 

void __threadfence(); waits until all global and shared memory accesses made by the calling thread prior to __threadfence() are visible to:  
> void __threadfence(); __threadfence() 이전에 호출한 스레드가 만든 모든 전역 및 공유 메모리 액세스가 다음에 표시될 때까지 대기합니다.

All threads in the thread block for shared memory accesses,  All threads in the device for global memory accesses. 
> 공유 메모리 액세스를 위한 스레드 블록의 모든 스레드, 전역 메모리 액세스를 위한 디바이스의 모든 스레드

void __threadfence_system(); waits until all global and shared memory accesses made by the calling thread prior to __threadfence_system() are visible to:  
> void __threadfence_system (); __threadfence_system() 이전의 호출 스레드가 만든 모든 전역 및 공유 메모리 액세스가 다음에 표시될 때까지 대기합니다.

All threads in the thread block for shared memory accesses,  All threads in the device for global memory accesses, Host threads for page-locked host memory accesses (see Section 3.2.4.3). 
> 공유 메모리 액세스를 위한 스레드 블록의 모든 스레드, 전역 메모리 액세스를 위한 디바이스의 모든 스레드, 페이지 잠금 호스트 메모리 액세스를 위한 호스트 스레드 (3.2.4.3 절 참조).
 
__threadfence_system() is only supported by devices of compute capability 2.x and higher. 
> __threadfence_system()은 컴퓨팅 기능이 2.x 이상인 디바이스에서만 지원됩니다.

In general, when a thread issues a series of writes to memory in a particular order, other threads may see the effects of these memory writes in a different order. 
> 일반적으로 스레드가 특정 순서로 메모리에 일련의 쓰기를 발행하면 다른 스레드는 이러한 메모리 쓰기의 효과를 다른 순서로 볼 수 있습니다.

__threadfence_block(), __threadfence(), and __threadfence_system() can be used to enforce some ordering. 
> __threadfence_block(), __threadfence() 및 __threadfence_system()을 사용하여 일부 순서를 적용할 수 있습니다.

One use case is when threads consume some data produced by other threads as illustrated by the following code sample of a kernel that computes the sum of an array of N numbers in one call. 
> 하나의 사용 사례는 하나의 호출에서 N 개의 숫자 배열 합계를 계산하는 커널의 다음 코드 샘플에서 설명하는 것처럼 다른 스레드에서 생성된 일부 데이터를 스레드가 소비하는 경우입니다.

Each block first sums a subset of the array and stores the result in global memory. 
> 각 블록은 먼저 배열의 하위 집합을 합하고 그 결과를 전역 메모리에 저장합니다.

When all blocks are done, the last block done reads each of these partial sums from global memory and sums them to obtain the final result. 
> 모든 블록이 완료되면 마지막 블록은 전역 메모리에서 이러한 부분 합계를 읽고 최종 결과를 얻기 위해 합계합니다.

In order to determine which block is finished last, each block atomically increments a counter to signal that it is done with computing and storing its partial sum (see Section B.11 about atomic functions).  
> 마지막으로 완료된 블록을 결정하기 위해 각 블록은 카운터를 원자적으로 증가시켜 부분 합계를 계산하고 저장하는 작업을 완료했음을 알립니다 (원자적 기능에 대해서는 B.11 절 참조).

The last block is the one that receives the counter value equal to gridDim.x-1. 
> 마지막 블록은 gridDim.x-1과 같은 카운터 값을 받는 블록입니다.

If no fence is placed between storing the partial sum and incrementing the counter, the counter might increment before the partial sum is stored and therefore, might reach gridDim.x-1 and let the last block start reading partial sums before they have been actually updated in memory.  
> 부분 합을 저장하고 카운터를 증가시키는 사이에 펜스가 없으면 부분 합계가 저장되기 전에 카운터가 증가하여 gridDim.x-1에 도달할 수 있고, 마지막 블록이 메모리에서 실제로 업데이트되기 전에 부분 합계를 읽기 시작합니다.

B.6 Synchronization Functions
> B.6 동기화 기능 

void __syncthreads(); waits until all threads in the thread block have reached this point and all global and shared memory accesses made by these threads prior to __syncthreads() are visible to all threads in the block. 
> void __syncthreads(); 스레드 블록의 모든 스레드가 이 지점에 도달할 때까지 대기하고 __syncthreads() 이전에 이러한 스레드가 만든 모든 전역 및 공유 메모리 액세스가 블록의 모든 스레드에서 볼 수 있을 때까지 대기합니다.

__syncthreads() is used to coordinate communication between the threads of the same block. 
> __syncthreads()는 동일한 블록의 스레드 간 통신을 조정하는 데 사용됩니다.

When some threads within a block access the same addresses in shared or global memory, there are potential read-after-write, write-after-read, or write-after-write hazards for some of these memory accesses. 
> 블록 내의 일부 스레드가 공유 또는 전역 메모리의 동일한 주소에 액세스할 때 이러한 메모리 액세스에 대해 잠재적인 쓰기 후 읽기, 읽기 후 쓰기 또는 쓰기 후 쓰기 위험이 있습니다.

These data hazards can be avoided by synchronizing threads in-between these accesses. 
> 이러한 데이터 위험은 이러한 액세스 간에 스레드를 동기화하여 방지할 수 있습니다. 

__syncthreads() is allowed in conditional code but only if the conditional evaluates identically across the entire thread block, otherwise the code execution is likely to hang or produce unintended side effects. 
> __syncthreads()는 조건부 코드에서 허용되지만 조건부가 전체 스레드 블록에서 동일하게 평가되는 경우에만 허용됩니다. 그렇지 않으면 코드 실행이 중지되거나 의도하지 않은 부작용이 발생할 수 있습니다.

Devices of compute capability 2.x and higher support three variations of __syncthreads() described below. 
> 컴퓨팅 기능이 2.x 이상인 디바이스는 아래에 설명된 __syncthreads()의 세 가지 변형을 지원합니다.

int __syncthreads_count(int predicate); is identical to __syncthreads() with the additional feature that it evaluates predicate for all threads of the block and returns the number of threads for which predicate evaluates to non-zero. 
> int __syncthreads_count(int predicate); 는 __syncthreads()와 동일하며 블록의 모든 스레드에 대한 술어를 평가하는 추가 기증이 있으며 술어가 0이 아닌 것으로 평가되는 스레드의 수를 반환합니다.

int __syncthreads_and(int predicate); is identical to __syncthreads() with the additional feature that it evaluates predicate for all threads of the block and returns non-zero if and only if predicate evaluates to non-zero for all of them. 
> int __syncthreads_and(int predicate); 는 __syncthreads()와 동일하며 블록의 모든 스레드에 대한 술어를 평가하는 추가 기능이 있으며 술어가 그 중 하나에 대해 0이 아닌 것으로 평가되는 경우에만 0이 아닌 값을 반환합니다.

int __syncthreads_or(int predicate); is identical to __syncthreads() with the additional feature that it evaluates predicate for all threads of the block and returns non-zero if and only if predicate evaluates to non-zero for any of them. 
> int __syncthreads_or (int predicate); 는 __syncthreads()와 동일하며 블록의 모든 스레드에 대한 술어를 평가하는 추가 기능이 있으며 술어가 그 중 하나에 대해 0이 아닌 것으로 평가되는 경우에만 0이 아닌 값을 반환합니다.

## B.7 Mathematical Functions 
> B.7 수학 함수

The reference manual lists all C/C++ standard library mathematical functions that are supported in device code and all intrinsic functions that are only supported in device code. 
> 참조 설명서에는 디바이스 코드에서 지원되는 모든 C/C++ 표준 라이브러리 수학 함수와 디바이스 코드에서만 지원되는 모든 내장 함수가 나와 있습니다.

Appendix C provides accuracy information for some of these functions when relevant. 
> 부록 C는 관련이 있을 때 이러한 함수 중 일부에 대한 정확도 정보를 제공합니다.

## B.8 Texture Functions 
> B.8 텍스처 함수

For texture functions, a combination of the texture reference’s immutable (i.e. compile-time) and mutable (i.e. runtime) attributes determine how the texture coordinates are interpreted, what processing occurs during the texture fetch, and the return value delivered by the texture fetch. Immutable attributes are described in Section 3.2.10.1.1. 
> 텍스처 함수의 경우, 텍스처 참조의 불변 (즉, 컴파일 타임)의 조합과 변경 가능한 (즉, 런타임) 속성의 조합은 텍스처 좌표가 어떻게 해석되는지, 텍스처 가져오기 중에 어떤 처리가 이루어지는지, 텍스처 가져오기에 의해 전달되는 반환값을 결정합니다. 변경할 수 없는 속성은 3.2.10.1.1 절에 설명되어 있습니다.

Mutable attributes are described in Section 3.2.10.1.2. Texture fetching is described in Appendix E. 
> 변경 가능한 속성은 3.2.10.1.2 절에 설명되어 있습니다. 텍스처 가져오기는 부록 E에 설명되어 있습니다.

## B.8.1 tex1Dfetch() 
> B.8.1 tex1Dfetch() 

fetch the region of linear memory bound to texture reference texRef using integer texture coordinate x. 
> 정수 텍스처 좌표 x를 사용하여 텍스처 참조 texRef에 바인딩된 선형 메모리 영역을 가져옵니다.

tex1Dfetch() only works with non-normalized coordinates (Section 3.2.10.1.2), so only the border and clamp addressing modes are supported. are supported. 
> tex1Dfetch()는 정규화되지 않은 좌표에서만 작동하므로 (3.2.10.1.2 절) 경계 및 클램프(고정) 주소 지정 모드만 지원됩니다. 지원됩니다.

It does not perform any texture filtering. 
> 텍스처 필터링을 수행하지는 않습니다.

For integer types, it may optionally promote the integer to single-precision floating point. Besides the functions shown above, 2-, and 4-tuples are supported; for example: 
> 정수 타입의 경우 정수를 단 정밀도 부동 소수점으로 선택적으로 승격시킬 수 있습니다. 위에 표시된 함수 외에도 2-, 4- 튜플이 지원됩니다. 예를 들면:

fetches the region of linear memory bound to texture reference texRef using texture coordinate x. 
> 텍스처 좌표 x를 사용하여 텍스처 참조 texRef에 바인딩된 선형 메모리 영역을 가져옵니다.

## B.8.2 tex1D() 
> B.8.2 tex1D() 

fetches the CUDA array bound to the one-dimensional texture reference texRef using texture coordinate x. 
> 텍스처 좌표 x를 사용하여 1 차원 텍스처 참조 texRef에 바인딩된 CUDA 배열을 가져옵니다.

## B.8.3 tex2D() 
> B.8.3 tex2D() 

fetches the CUDA array or the region of linear memory bound to the two-dimensional texture reference texRef using texture coordinates x and y. 
> 텍스처 좌표 x와 y를 사용하여 2 차원 텍스처 참조 texRef에 바인딩된 CUDA 배열 또는 선형 메모리 영역을 가져옵니다.

## B.8.4 tex3D() 
> B.8.4 tex3D() 

fetches the CUDA array bound to the three-dimensional texture reference texRef using texture coordinates x, y, and z. 
> 텍스처 좌표 x, y, z를 사용하여 3 차원 텍스처 참조 texRef에 바인딩된 CUDA 배열을 가져옵니다.

## B.8.5 tex1DLayered() 
> B.8.5 tex1DLayered() 

fetches the CUDA array bound to the one-dimensional layered texture reference texRef using texture coordinate x and index layer, as described in Section 3.2.10.1.5. 
> 3.2.10.1.5 절에서 설명한 대로 텍스처 좌표 x와 인덱스 레이어를 사용하여 1 차원 계층화된 텍스처 참조 texRef에 바인딩된 CUDA 배열을 가져옵니다.

## B.8.6 tex2DLayered() 
> B.8.6 tex2DLayered() 

fetches the CUDA array bound to the two-dimensional layered texture reference texRef using texture coordinates x and y, and index layer, as described in Section 3.2.10.1.5. 
> 3.2.10.1.5 절에서 설명한 대로 텍스처 좌표 x와 y와 인덱스 레이어를 사용하여 2 차원 계층화된 텍스처 참조 texRef에 바인딩된 CUDA 배열을 가져옵니다.

## B.8.7 texCubemap() 
> B.8.7 texCubemap() 

fetches the CUDA array bound to the cubemap texture reference texRef using texture coordinates x, y, and z, as described in Section 3.2.10.1.6. 
> 3.2.10.1.6 절에서 설명한 대로 텍스처 좌표 x, y 및 z를 사용하여 큐브맵 텍스처 참조 texRef에 바인딩된 CUDA 배열을 가져옵니다.

## B.8.8 texCubemapLayered() 
> B.8.8 texCubemapLayered() 

fetches the CUDA array bound to the cubemap layered texture reference texRef using texture coordinates x, y, and z, and index layer, as described in Section 3.2.10.1.7. 
> 3.2.10.1.7 절에서 설명한 대로 텍스처 좌표 x, y, z와 인덱스 레이어를 사용하여 큐브맵 계층화된 텍스처 참조 texRef에 바인딩된 CUDA 배열을 가져옵니다.

## B.8.9 tex2Dgather() 
> B.8.9 tex2Dgather() 

fetches the CUDA array bound to the cubemap texture reference texRef using texture coordinates x and y, as described in Section 3.2.10.1.8. 
> 3.2.10.1.8 절에서 설명한 대로 텍스처 좌표 x와 y를 사용하여 큐브맵 텍스처 참조 texRef에 바인딩된 CUDA 배열을 가져옵니다.

B.9 Surface Functions 
> B.9 표면 함수

Surface functions are only supported by devices of compute capability 2.0 and higher. 
> 표면 함수는 컴퓨팅 기능이 2.0 이상인 디바이스에서만 지원됩니다.

Surface reference declaration is described in Section 3.2.10.2.1 and surface binding in Section 3.2.10.2.2. 
> 표면 참조 선언은 3.2.10.2.1 절에 설명되어 있으며 표면 바인딩은 3.2.10.2.2 절에 설명되어 있습니다.

In the sections below, boundaryMode specifies the boundary mode, that is how out-of-range surface coordinates are handled; 
> 아래 섹션에서 boundaryMode는 경계 모드를 지정합니다. 즉, 범위를 벗어난 표면 좌표가 처리되는 방법입니다.

it is equal to either cudaBoundaryModeClamp, in which case out-of-range coordinates are clamped to the valid range, or cudaBoundaryModeZero, in which case out-of-range reads return zero and out-of-range writes are ignored, 
> 이것은 cudaBoundaryModeClamp와 같습니다. 이 경우 범위를 벗어나는 좌표는 유효한 범위로 고정됩니다. 또는 cudaBoundaryModeZero, 이 경우 범위를 벗어나는 읽기는 0을 반환하고 범위를 벗어난 쓰기는 무시됩니다,

or cudaBoundaryModeTrap, in which case out-of-range accesses cause the kernel execution to fail. 
> 또는 cudaBoundaryModeTrap, 이 경우 범위를 벗어난 액세스로 인해 커널 실행이 실패하게 됩니다.

## B.9.1 surf1Dread() 
> B.9.1 surf1Dread() 

reads the CUDA array bound to the one-dimensional surface reference surfRef using coordinate x. 
> 좌표 x를 사용하여 1 차원 표면 참조 surfRef에 바인딩된 CUDA 배열을 읽습니다.

## B.9.2 surf1Dwrite() 
> B.9.2 surf1Dwrite() 

writes value data to the CUDA array bound to the one-dimensional surface reference surfRef at coordinate x. 
> 좌표 x에서 1 차원 표면 참조 surfRef에 바인딩된 CUDA 배열에 값 데이터를 씁니다.

## B.9.3 surf2Dread() 
> B.9.3 surf2Dread() 

reads the CUDA array bound to the two-dimensional surface reference surfRef using coordinates x and y. 
> 좌표 x와 y를 사용하여 2 차원 표면 참조 surfRef에 바인딩된 CUDA 배열을 읽습니다.

## B.9.4 surf2Dwrite() 
> B.9.4 surf2Dwrite() 

writes value data to the CUDA array bound to the two-dimensional surface reference surfRef at coordinate x and y. 
> 좌표 x와 y에서 2 차원 표면 참조 surfRef에 바인딩된 CUDA 배열에 값 데이터를 씁니다.

## B.9.5 surf3Dread() 
> B.9.5 surf3Dread() 

reads the CUDA array bound to the three-dimensional surface reference surfRef using coordinates x, y, and z. 
> 좌표 x, y 및 z를 사용하여 3 차원 표면 참조 surfRef에 바인딩된 CUDA 배열을 읽습니다.

## B.9.6 surf3Dwrite() 
> B.9.6 surf3Dwrite() 

writes value data to the CUDA array bound to the three-dimensional surface reference surfRef at coordinate x, y, and z. 
> 좌표 x, y 및 z에서 3 차원 표면 참조 surfRef에 바인딩된 CUDA 배열에 값 데이터를 씁니다.

## B.9.7 surf1DLayeredread() 
> B.9.7 surf1DLayeredread() 

reads the CUDA array bound to the one-dimensional layered surface reference surfRef using coordinate x and index layer. 
> 좌표 x와 인덱스 레이어를 사용하여 1 차원 계층화된 표면 참조 surfRef에 바인딩된 CUDA 배열을 읽습니다.

## B.9.8 surf1DLayeredwrite() 
> B.9.8 surf1DLayeredwrite() 

writes value data to the CUDA array bound to the two-dimensional layered surface reference surfRef at coordinate x and index layer. 
> 좌표 x와 인덱스 레이어에서 2 차원 계층화된 표면 참조 surfRef에 바인딩된 CUDA 배열에 값 데이터를 씁니다.

## B.9.9 surf2DLayeredread() 
> B.9.9 surf2DLayeredread() 

reads the CUDA array bound to the two-dimensional layered surface reference surfRef using coordinate x and y, and index layer. 
> 좌표 x 및 y와 인덱스 레이어를 사용하여 2 차원 계층화된 표면 참조 surfRef에 바인딩된 CUDA 배열을 읽습니다.

## B.9.10 surf2DLayeredwrite() 
> B.9.10 surf2DLayeredwrite() 

writes value data to the CUDA array bound to the one-dimensional layered surface reference surfRef at coordinate x and y, and index layer. 
> 좌표 x 및 y와 인덱스 레이어에서 1 차원 계층화된 표면 참조 surfRef에 바인딩된 CUDA 배열에 값 데이터를 씁니다.

## B.9.11 surfCubemapread() 
> B.9.11 surfCubemapread() 

reads the CUDA array bound to the cubemap surface reference surfRef using coordinate x and y, and face index face. 
> 좌표 x 및 y와 면 인덱스 면을 사용하여 큐브맵 표면 참조 surfRef에 바인딩된 CUDA 배열을 읽습니다.

## B.9.12 surfCubemapwrite() 
> B.9.12 surfCubemapwrite() 

writes value data to the CUDA array bound to the cubemap reference surfRef at coordinate x and y, and face index face. 
> 좌표 x 및 y와 면 인덱스 면에서 큐브맵 참조 surfRef에 바인딩된 CUDA 배열에 값 데이터를 씁니다.

## B.9.13 surfCubemabLayeredread() 
> B.9.13 surfCubemabLayeredread() 

reads the CUDA array bound to the cubemap layered surface reference surfRef using coordinate x and y, and index layerFace. 
> 좌표 x 및 y와 인덱스 layerFace를 사용하여 큐브맵 계층화된 표면 참조 surfRef에 바인딩된 CUDA 배열을 읽습니다.

## B.9.14 surfCubemapLayeredwrite() 
> B.9.14 surfCubemapLayeredwrite() 

writes value data to the CUDA array bound to the cubemap layered reference surfRef at coordinate x and y, and index layerFace. 
> 좌표 x 및 y와 인덱스 layerFace에서 큐브맵 계층화된 참조 surfRef에 바인딩된 CUDA 배열에 값 데이터를 씁니다.

## B.10 Time Function 
> B.10 시간 기능

when executed in device code, returns the value of a per-multiprocessor counter that is incremented every clock cycle. 
> 디바이스 코드에서 실행될 때 시간 사이클마다 증가되는 다중프로세서 당 카운터의 값을 반환합니다.

Sampling this counter at the beginning and at the end of a kernel, taking the difference of the two samples, and recording the result per thread provides a measure for each thread of the number of clock cycles taken by the device to completely execute the thread, but not of the number of clock cycles the device actually spent executing thread instructions. 
> 이 카운터를 커널의 시작과 끝에서 샘플링하고 두 샘플의 차이를 취한 다음, 스레드 당 결과를 기록한 것은 스레드가 완전히 실행하도록 디바이스로 얻은 시간 사이클 수의 각  스레드에 대한 측정값을 제공하지만 디바이스가 실제로 스레드 명령을 실행하는 데 소비한 시간 사이클 수는 아닙니다.

The former number is greater than the latter since threads are time sliced. 
> 이전 수는 쓰레드가 시간 분할되어 있기 때문에 후자보다 큽니다.

## B.11 Atomic Functions 
> B.11 원자 함수

An atomic function performs a read-modify-write atomic operation on one 32-bit or 64-bit word residing in global or shared memory. 
> 원자 함수는 전역 또는 공유 메모리에 있는 하나의 32 비트 또는 64 비트 단어에서 읽기-수정-쓰기 원자 연산을 수행합니다.

For example, atomicAdd() reads a 32-bit word at some address in global or shared memory, adds a number to it, and writes the result back to the same address. 
> 예를 들어, atomicAdd()는 전역 또는 공유 메모리의 일부 주소에서 32 비트 단어를 읽고 숫자를 추가한 다음 결과를 동일한 주소에 다시 씁니다.

The operation is atomic in the sense that it is guaranteed to be performed without interference from other threads. 
> 작업은 다른 스레드의 간섭없이 수행된다는 점에서 원자적입니다.

In other words, no other thread can access this address until the operation is complete. 
> 즉, 작업이 완료 될 때까지 다른 스레드가 이 주소에 액세스할 수 없습니다.

Atomic functions can only be used in device functions and atomic functions operating on mapped page-locked memory (Section 3.2.4.3) are not atomic from the point of view of the host or other devices. 
> 원자 함수는 디바이스 기능에서만 사용될 수 있으며 매핑된 페이지 잠금 메모리 ( 3.2.4.3 절)에서 작동하는 원자 함수는 호스트 또는 다른 디바이스의 관점에서 볼 때 원자적이지 않습니다.

As mentioned in Table F-1, the support for atomic operations varies with the compute capability: 
> 표 F-1에서 언급했듯이 원자 연산 지원은 컴퓨팅 기능에 따라 다릅니다.

Atomic functions are only available for devices of compute capability 1.1 and higher.  
> 원자 함수는 컴퓨팅 기능 1.1 이상의 디바이스에서만 사용할 수 있습니다.

Atomic functions operating on 32-bit integer values in shared memory and atomic functions operating on 64-bit integer values in global memory are only available for devices of compute capability 1.2 and higher.  
> 공유 메모리의 32 비트 정수 값에서 작동하는 원자 함수와 전역 메모리의 64 비트 정수 값에서 작동하는 원자 함수는 컴퓨팅 기능이 1.2 이상인 디바이스에서만 사용할 수 있습니다.

Atomic functions operating on 64-bit integer values in shared memory are only available for devices of compute capability 2.x and higher.  
> 공유 메모리의 64 비트 정수 값에서 작동하는 원자 함수는 컴퓨팅 기능 2.x 이상의 디바이스에서만 사용할 수 있습니다.

Only atomicExch() and atomicAdd() can operate on 32-bit floating-point values:  in global memory for atomicExch() and devices of compute capability 1.1 and higher.  
> atomicExch() 및 atomicAdd()만 32 비트 부동 소수점 값에서 작동할 수 있습니다. atomicExch()의 전역 메모리 및 컴퓨팅 기능 1.1 이상의 디바이스에서.

in shared memory for atomicExch() and devices of compute capability 1.2 and higher.  
> atomicExch() 및 컴퓨팅 기능 1.2 이상의 디바이스에 대한 공유 메모리.

in global and shared memory for atomicAdd() and devices of compute capability 2.x and higher. 
> atomicAdd()에 대한 전역 및 공유 메모리 및 컴퓨팅 기능이 2.x 이상인 디바이스에서.

Note however that any atomic operation can be implemented based on atomicCAS() (Compare And Swap). 
> 그러나 원자 연산은 atomicCAS() (Compare And Swap)에 기반하여 구현될 수 있습니다.

For example, atomicAdd() for double-precision floating-point numbers can be implemented as follows: 
> 예를 들어, 배 정밀도 부동 소수점 수의 경우 atomicAdd()는 다음과 같이 구현할 수 있습니다.

## B.11.1 Arithmetic Functions 
> B.11.1 산술 함수

## B.11.1.1 atomicAdd() int atomicAdd(int* address, int val); 
> B.11.1.1 atomicAdd() int atomicAdd(int* address, int val); 

float atomicAdd(float* address, float val); reads the 32-bit or 64-bit word old located at the address address in global or shared memory, computes (old + val), and stores the result back to memory at the same address. 
> float atomicAdd (float * address, float val); 전역 또는 공유 메모리의 주소 주소에 있는 32 비트 또는 64 비트 단어를 읽고, (old + val)을 계산하고 그 결과를 같은 주소의 메모리에 다시 저장합니다.

These three operations are performed in one atomic transaction. 
> 이 세 가지 작업은 하나의 원자 트랜잭션으로 수행됩니다.

The function returns old. The floating-point version of atomicAdd() is only supported by devices of compute capability 2.x and higher. 
> 이 함수는 old를 반환합니다. atomicAdd()의 부동 소수점 버전은 컴퓨팅 기능이 2.x 이상인 디바이스에서만 지원됩니다.

## B.11.1.2 atomicSub() int atomicSub(int* address, int val);  
> B.11.1.2 atomicSub() int atomicSub(int* address, int val);  

reads the 32-bit word old located at the address address in global or shared memory, computes (old - val), and stores the result back to memory at the same address. 
> 전역 또는 공유 메모리의 주소 주소에 있는 32 비트 단어 old를 읽고 계산 (old - val) 하고 그 결과를 동일한 주소의 메모리에 다시 저장합니다.

These three operations are performed in one atomic transaction. The function returns old. 
> 이 세 가지 작업은 하나의 원자 트랜잭션으로 수행됩니다. 이 함수는 old를 반환합니다.

## B.11.1.3 atomicExch() int atomicExch(int* address, int val); 
> B.11.1.3 atomicExch() int atomicExch(int* address, int val); 

reads the 32-bit or 64-bit word old located at the address address in global or shared memory and stores val back to memory at the same address. 
> 전역 또는 공유 메모리의 주소 주소에 있는 32 비트 또는 64 비트 단어 old를 읽고 val을 동일한 주소의 메모리에 다시 저장합니다.

These two operations are performed in one atomic transaction. The function returns old. 
> 이 두 연산은 하나의 원자 트랜잭션으로 수행됩니다. 이 함수는 old를 반환합니다.

## 11.1.4 atomicMin() int atomicMin(int* address, int val); 
> 11.1.4 atomicMin() int atomicMin(int* address, int val); 

reads the 32-bit word old located at the address address in global or shared memory, computes the minimum of old and val, and stores the result back to memory at the same address. 
> 전역 또는 공유 메모리에서 주소 주소에있는 32 비트 단어 old를 읽고 old 및 val의 최소값을 계산하고, 그 결과를 동일한 주소의 메모리에 다시 저장합니다.

These three operations are performed in one atomic transaction. The function returns old.
> 이 세 가지 작업은 하나의 원자 트랜잭션으로 수행됩니다. 이 함수는 old를 반환합니다.

## B.11.1.5 atomicMax() int atomicMax(int* address, int val); 
> B.11.1.5 atomicMax() int atomicMax(int* address, int val); 

reads the 32-bit word old located at the address address in global or shared memory, computes the maximum of old and val, and stores the result back to memory at the same address. 
> 전역 또는 공유 메모리에서 주소 주소에있는 32 비트 단어 old를 읽고 old 및 val의 최대값을 계산한 다음 그 결과를 동일한 주소의 메모리에 다시 저장합니다.

These three operations are performed in one atomic transaction. The function returns old.
> 이 세 가지 작업은 하나의 원자 트랜잭션으로 수행됩니다. 이 함수는 old를 반환합니다.

## B.11.1.6 atomicInc() unsigned int atomicInc(unsigned int* address,
> B.11.1.6 atomicInc() unsigned int atomicInc(unsigned int* address,

reads the 32-bit word old located at the address address in global or shared memory, computes ((old >= val) ? 0 : (old+1)), and stores the result back to memory at the same address. 
> 전역 또는 공유 메모리의 주소 주소에있는 32 비트 단어 old를 읽고, (old> = val)? 0 : (old + 1)을 계산하고 그 결과를 동일한 주소의 메모리에 다시 저장합니다.

These three operations are performed in one atomic transaction. The function returns old. 
> 이 세 가지 작업은 하나의 원자 트랜잭션으로 수행됩니다. 이 함수는 old를 반환합니다.

## B.11.1.7 atomicDec() unsigned int atomicDec(unsigned int* address,   unsigned int val); 
> B.11.1.7 atomicDec() unsigned int atomicDec(unsigned int* address,   unsigned int val); 

reads the 32-bit word old located at the address address in global or shared memory, computes (((old == 0) | (old > val)) ? val : (old-1)), and stores the result back to memory at the same address. 
> 전역 또는 공유 메모리의 주소 주소에있는 32 비트 단어 old를 읽고, ((old == 0) | (old> val)) val : (old-1))을 계산하고 그 결과를 동일한 주소의 메모리에 다시 저장합니다..

These three operations are performed in one atomic transaction. The function returns old. 
> 이 세 가지 작업은 하나의 원자 트랜잭션으로 수행됩니다. 이 함수는 old를 반환합니다.
 
## B.11.1.8 atomicCAS() int atomicCAS(int* address, int compare, int val); 
> B.11.1.8 atomicCAS() int atomicCAS(int* address, int compare, int val); 

reads the 32-bit or 64-bit word old located at the address address in global or shared memory, computes (old == compare ? val : old), and stores the result back to memory at the same address. 
> 전역 또는 공유 메모리의 주소 주소에있는 32 비트 또는 64 비트 단어를 읽고 계산 (old == compare? val : old)하고, 그 결과를 동일한 주소의 메모리에 다시 저장합니다.

These three operations are performed in one atomic transaction. The function returns old (Compare And Swap). 
> 이 세 가지 작업은 하나의 원자 트랜잭션으로 수행됩니다. 이 함수는 old (Compare And Swap)를 반환합니다.

## B.11.2 Bitwise Functions 
> B.11.2 비트 단위 함수

## B.11.2.1 atomicAnd()  
> B.11.2.1 atomicAnd()  

reads the 32-bit word old located at the address address in global or shared memory, computes (old & val), and stores the result back to memory at the same address. 
> 전역 또는 공유 메모리의 주소 주소에있는 32 비트 단어 old를 읽고 계산 (old 및 val) 하고, 그 결과를 동일한 주소의 메모리에 다시 저장합니다.

These three operations are performed in one atomic transaction. The function returns old.
> 이 세 가지 작업은 하나의 원자 트랜잭션으로 수행됩니다. 이 함수는 old를 반환합니다.

## B.11.2.2 atomicOr() 
> B.11.2.2 atomicOr() 

reads the 32-bit word old located at the address address in global or shared memory, computes (old | val), and stores the result back to memory at the same address. 
> 전역 또는 공유 메모리의 주소 주소에 있는 32 비트 단어 old를 읽고 계산 (old | val) 하고, 그 결과를 동일한 주소의 메모리에 다시 저장합니다.

These three operations are performed in one atomic transaction. The function returns old.
> 이 세 가지 작업은 하나의 원자 트랜잭션으로 수행됩니다. 이 함수는 old를 반환합니다.

## B.11.2.3 atomicXor() 
> B.11.2.3 atomicXor() 
 
reads the 32-bit word old located at the address address in global or shared memory, computes (old ^ val), and stores the result back to memory at the same address. 
> 전역 또는 공유 메모리의 주소 주소에 있는 32 비트 단어 old를 읽고 계산 (old ^ val)하고,
그 결과를 동일한 주소의 메모리에 다시 저장합니다.

These three operations are performed in one atomic transaction. The function returns old. 
> 이 세 가지 작업은 하나의 원자 트랜잭션으로 수행됩니다. 이 함수는 old를 반환합니다.
 
## B.12 Warp Vote Functions 
> B.12 워프 투표 기능

Warp vote functions are only supported by devices of compute capability 1.2 and higher (see Section 4.1 for the definition of a warp). 
> 워프 투표 기능은 컴퓨팅 기능 1.2 이상의 디바이스에서만 지원됩니다 (워프 정의에 대해서는 4.1 절 참조).

int __all(int predicate); evaluates predicate for all active threads of the warp and returns non-zero if and only if predicate evaluates to non-zero for all of them. 
> int __all(int predicate); 워프의 모든 활성 스레드에 대한 술어를 평가하고 술어가 모두 0이 아닌 것으로 평가하는 경우에만 0이 아닌 값을 반환합니다.

int __any(int predicate); evaluates predicate for all active threads of the warp and returns non-zero if and only if predicate evaluates to non-zero for any of them. 
> int __any(int predicate); 워프의 모든 활성 스레드에 대한 술어를 평가하고 술어가 그 중 하나에 대해 0이 아닌 것으로 평가되는 경우에만 0이 아닌 값을 반환합니다.

unsigned int __ballot(int predicate); evaluates predicate for all active threads of the warp and returns an integer whose Nth bit is set if and only if predicate evaluates to non-zero for the Nth thread of the warp. 
> unsigned int __ballot (int predicate); 워프의 모든 활성 스레드에 대한 술어를 평가하고 워프의 N 번째 스레드에 대해 술어가 0이 아닌 것으로 평가되는 경우에만 N 번째 비트가 설정된 정수를 반환합니다.

This function is only supported by devices of compute capability 2.x and higher. 
> 이 기능은 컴퓨팅 기능이 2.x 이상인 디바이스에서만 지원됩니다.

## B.13 Warp Shuffle Functions 
> B.13 워프 셔플 기능

__shfl, __shfl_up, __shfl_down, __shfl_xor exchange a variable between threads within a warp. 
> __shfl, __shfl_up, __shfl_down, __shfl_xor는 워프 내의 스레드 간에 변수를 교환합니다.

They are only supported by devices of compute capability 3.0 (see Section 4.1 for the definition of a warp). 
> 이것들은 컴퓨팅 기능 3.0의 디바이스들에 의해서만 지원됩니다 (워프의 정의에 대해서는 4.1 절을 참조).

## B.13.2 Description 
> B.13.2 설명

The __shfl() intrinsics permit exchanging of a variable between threads within a warp without use of shared memory. 
> __shfl() 내장 함수를 사용하면 공유 메모리를 사용하지 않고 워프 내의 스레드 간에 변수를 교환할 수 있습니다.

The exchange occurs simultaneously for all active threads within the warp, moving 4 bytes of data per thread. 
> 워프 내의 모든 활성 스레드에 대해 교환이 동시에 발생하여 스레드 당 4 바이트의 데이터가 이동합니다.

Exchange of 8byte quantities must be broken into two separate invocations of __shfl(). 
> 8 바이트 수량의 교환은 __shfl()의 두 가지 개별 호출로 분리되어야 합니다.
 
Threads within a warp are referred to as lanes, and for devices of compute capability 3.0 may have an index between 0 and 31 (inclusive). 
> 워프 내의 쓰레드는 레인 (lanes)으로 불리며, 컴퓨팅 기능 3.0의 디바이스는 0과 31(포함) 사이의 인덱스를 가질 수 있습니다.

Four source-lane addressing modes are supported: 
> 네 가지 소스 레인 주소 지정 모드가 지원됩니다. 

 __shfl(): Direct copy from indexed lane  __shfl_up(): 
> __shfl(): 인덱싱된 레인에서 직접 복사합니다 __shfl_up() :

Copy from a lane with lower ID relative to caller  __shfl_down(): Copy from a lane with higher ID relative to caller  __shfl_xor(): 
> __shfl_down() 발신자와 관련하여 ID가 낮은 레인에서 복사합니다 : __shfl_xor() 발신자와 관련하여 ID가 높은 차선에서 복사합니다 :

Copy from a lane based on bitwise XOR of own lane ID 
> 자신의 레인 ID의 비트 XOR을 기반으로 하는 레인에서 복사합니다.

Threads may only read data from another thread which is actively participating in the __shfl() command.
> 스레드는 __shfl() 명령에 적극적으로 참여하는 다른 스레드의 데이터만 읽을 수 있습니다.

If the target thread is inactive, the retrieved value is undefined. 
> 타겟 스레드가 비활성적인 경우, 검색된 값은 정의되지 않습니다.

All the  __shfl() intrinsics take an optional width parameter which permits subdivision of the warp into segments – for example to exchange data between 4 groups of 8 lanes in a SIMD manner. 
> 모든 __shfl() 내장 함수는 선택적 폭 매개 변수를 사용하여 워프를 세그먼트로 분할할 수 있습니다. 예를 들어 SIMD 방식으로 8 레인의 4 개 그룹 사이에서 데이터를 교환할 수 있습니다.

If width is less than 32 then each subsection of the warp behaves as a separate entity with a starting logical lane ID of 0. 
> 너비가 32보다 작으면 워프의 각 서브섹션은 시작 논리 레인 ID가 0인 별도의 엔티티로 동작합니다.

A thread may only exchange data with others in its own subsection. width must have a value which is a power of 2 so that the warp can be subdivided equally; 
> 스레드는 자체 서브섹션의 데이터만 교환할 수 있습니다. 너비는 2의 거듭제곱인 값을 가져야만 워프가 똑같이 세분될 수 있습니다.

results are undefined if width is not a power of 2, or is a number greater than warpSize. 
> 너비가 2의 거듭제곱이 아니면 warpSize보다 큰 숫자는 결과가 정의되지 않습니다. 

__shfl() returns the value of var held by the thread whose ID is given by srcLane. 
> __shfl()은 ID가 srcLane에 의해 지정된 스레드가 보유한 var의 값을 반환합니다.

If srcLane is outside the range [0:width-1], then the thread’s own value of var is returned. 
> srcLane이 [0:width-1]의 범위 외의 경우, 스레드 자체 var 값은 반환됩니다. 

__shfl_up() calculates a source lane ID by subtracting delta from the caller’s lane ID. 
> __shfl_up()은 호출자의 레인 ID에서 델타를 뺀 소스 레인 ID를 계산합니다.

The value of var held by the resulting lane ID is returned: in effect, var is shifted up the warp by delta lanes. 
> 결과 레인 ID에 포함된 var의 값은 반환됩니다. 실제로, var는 델타 레인 만큼 워프를 위로 이동합니다.

The source lane index will not wrap around the value of width, so effectively the lower delta lanes will be unchanged.
> 소스 레인 인덱스는 너비의 값을 랩(감싸지)하지 않으므로 효과적으로 낮은 델타 레인은 변경되지 않습니다.

 __shfl_down() calculates a source lane ID by adding delta to the caller’s lane ID. 
> __shfl_down()은 호출자의 레인 ID에 델타를 추가하여 소스 레인 ID를 계산합니다.

The value of var held by the resulting lane ID is returned: this has the effect of shifting var down the warp by delta lanes. 
> 결과 레인 ID에 의해 유지되는 var의 값이 반환됩니다. 이것은 var를 델타 레인으로 이동시키는 효과가 있습니다.

As for __shfl_up(), the ID number of the source lane will not wrap around the value of width and so the upper delta lanes will remain unchanged. 
> __shfl_up()의 경우 소스 레인의 ID 번호가 너비 값을 감싸지 않으므로 위쪽 델타 레인은 변경되지 않습니다.

__shfl_xor() calculates a source line ID by performing a bitwise XOR of the caller’s lane ID with laneMask: the value of var held by the resulting lane ID is returned. 
> __shfl_xor()는 호출자의 레인 ID와 laneMask의 비트 XOR을 수행하여 소스행 ID를 계산합니다. 결과 레인 ID가 보유한 var 값이 반환됩니다.

If the resulting lane ID falls outside the range permitted by width, the thread’s own value of var is returned. 
> 결과 레인 ID가 너비로 허용되는 범위를 벗어나면 스레드 자체 var 값이 반환됩니다.

This mode implements a butterfly addressing pattern such as is used in tree reduction and broadcast. 
> 이 모드는 트리 감소 및 브로드캐스트에 사용되는 버터플라이 주소 지정 패턴을 구현합니다.

## B.13.3 Return Value
> B.13.3 반환 값
 
All __shfl() intrinsics return the 4-byte word referenced by var from the source lane ID as an unsigned integer. 
> 모든 __shfl() 내장 함수는 var가 참조하는 4 바이트 단어를 소스 레인 ID에서 부호없는 정수로 반환합니다.

If the source lane ID is out of range or the source thread has exited, the calling thread’s own var is returned. 
> 소스 레인 ID가 범위를 벗어나거나 소스 스레드가 종료되면 호출 스레드의 자체 var가 반환됩니다.

## B.13.4 Notes 
> B.13.4 노트(메모)

All __shfl() intrinsics share the same semantics with respect to code motion as the vote intrinsics __any() and __all(). 
> 모든 __shfl() 내장 함수는 투표 내장 함수 __any() 및 __all()과 같은 코드 동작과 동일한 의미를 공유합니다.
 
Threads may only read data from another thread which is actively participating in the __shfl() command. 
> 스레드는 __shfl() 명령에 적극적으로 참여하고 있는 다른 스레드의 데이터만 읽을 수 있습니다.

If the target thread is inactive, the retrieved value is undefined. 
> 타겟 스레드가 비활성인 경우, 검색된 값은 정의되지 않습니다.

width must be a power-of-2 (i.e. 2, 4, 8, 16 or 32). Results are unspecified for other values. 
> 너비는 2의 제곱수 (예 : 2, 4, 8, 16 또는 32) 여야합니다. 다른 값에 대해서는 결과가 지정되지 않습니다.

Types other than int or float must first be cast in order to use the __shfl() intrinsics. 
> int 또는 float 이외의 유형은 먼저 __shfl() 내장 함수를 사용하기 위해 캐스팅(변환)되어야 합니다.

## B.13.5 Examples 
> B.13.5 예제

## B.13.5.1 Broadcast of a single value across a warp 
> B.13.5.1 워프 전역 단일 값 브로드캐스트

## B.13.5.2 Inclusive plus-scan across sub-partitions of 8 threads
> B.13.5.2  8개 스레드의 하위 분할 전역에 포괄적인 플러스 스캔
 
## B.14 Profiler Counter Function 
> B.14 프로파일러 카운터 기능

Each multiprocessor has a set of sixteen hardware counters that an application can increment with a single instruction by calling the __prof_trigger() function. 
> 각 다중프로세서에는 애플리케이션에서 __prof_trigger() 함수를 호출하여 단일 명령으로 증가시킬 수 있는 16 개의 하드웨어 카운터 집합이 있습니다.

void __prof_trigger(int counter); increments by one per warp the per-multiprocessor hardware counter of index counter. 
> void __prof_trigger(int counter); 인덱스 카운터의 다중프로세서 하드웨어 카운터 당 워프 당 하나씩 증가합니다.

Counters 8 to 15 are reserved and should not be used by applications. 
> 카운터 8 ~ 15는 예약되어 있으므로 애플리케이션에서 사용하면 안됩니다.

The value of counters 0, 1, …, 7 for the first multiprocessor can be obtained via the CUDA profiler by listing prof_trigger_00,  prof_trigger_01,  …, prof_trigger_07, etc. 
> 첫 번째 다중프로세서에 대한 카운터 0, 1, ..., 7의 값은 prof_trigger_00, prof_trigger_01, ..., prof_trigger_07 등을 나열하여 CUDA 프로파일러를 통해 얻을 수 있습니다.

in the profiler.conf file (see the profiler manual for more details). 
> profiler.conf 파일에 있습니다 (자세한 내용은 프로파일러 설명서 참조).

All counters are reset before each kernel call (note that when an application is run via cuda-gdb, the Visual Profiler, or the Parallel Nsight CUDA Debugger, all launches are synchronous). 
> 모든 카운터는 각 커널이 호출되기 전에 재설정됩니다 (애플리케이션이 cuda-gdb, Visual Profiler 또는 Parallel Nsight CUDA Debugger를 통해 실행될 때 모든 시작은 동기식입니다).

## B.15 Assertion 
> B.15 어설션(주장)

Assertion is only supported by devices of compute capability 2.x and higher. 
> 어설션(주장)은 컴퓨팅 기능이 2.x 이상인 디바이스에서만 지원됩니다. 

void assert(int expression); stops the kernel execution if expression is equal to zero. 
> void assert(int expression); 표현식이 0과 같은 경우 커널 실행을 중지합니다.

If the program is run within a debugger, this triggers a breakpoint and the debugger can be used to inspect the current state of the device. 
> 디버거 내에서 프로그램을 실행하면 중단점이 표시되고 디버거를 사용하여 디바이스의 현재 상태를 검사할 수 있습니다.

Otherwise, each thread for which expression is equal to zero prints a message to stderr after synchronization with the host via cudaDeviceSynchronize(), cudaStreamSynchronize(), or cudaEventSynchronize(). 
> 그렇지 않으면 표현식이 0과 같은 각 스레드는 cudaDeviceSynchronize(), cudaStreamSynchronize() 또는 cudaEventSynchronize()를 통해 호스트와 동기화된 후 stderr에 메시지를 출력합니다.

The format of this message is as follows: 
> 이 메시지의 형식은 다음과 같습니다.
 
Any subsequent host-side synchronization calls made for the same device will return cudaErrorAssert. 
> 동일한 디바이스에 대해 수행된 후속 호스트 측 동기화 호출은 cudaErrorAssert를 반환합니다.

No more commands can be sent to this device until cudaDeviceReset() is called to reinitialize the device. 
> cudaDeviceReset()이 호출되어 디바이스를 다시 초기화할 때까지 이 디바이스로 더 이상 명령을 보낼 수 없습니다.

If expression is different from zero, the kernel execution is unaffected. 
> 표현식이 0과 다른 경우, 커널 실행은 영향을 받지 않습니다.

For example, the following program from source file test.cu #include <assert.h> 
> 예를 들어, 소스 파일 test.cu의 다음 프로그램은 #include <assert.h>

Assertions are for debugging purposes. 
> 어설션은 디버깅을 목적으로 합니다.

They can affect performance and it is therefore recommended to disable them in production code. 
> 성능에 영향을 줄 수 있으므로 프로덕션 코드에서 사용하지 않도록 설정하는 것이 좋습니다.

They can be disabled at compile time by defining the NDEBUG preprocessor macro before including assert.h. 
> assert.h를 포함하기 전에 NDEBUG 전프로세서 매크로를 정의하여 컴파일 타임에 비활성화 할 수 있습니다.

Note that expression should not be an expression with side effects (something like (++i > 0), for example), otherwise disabling the assertion will affect the functionality of the code. 
> 표현식은 부작용이 있는 표현식 (예 : (++ i> 0)과 같은 것)이 아니어야 합니다. 그렇지 않으면 어설션을 비활성화하면 코드의 기능에 영향을 미칩니다.

## B.16 Formatted Output 
> B.16 형식화된 출력

Formatted output is only supported by devices of compute capability 2.x and higher. 
> 형식화된 출력은 컴퓨팅 기능이 2.x 이상인 디바이스에서만 지원됩니다.

int printf(const char *format[, arg, ...]); prints formatted output from a kernel to a host-side output stream. 
> int printf(const char *format[, arg, ...]); 커널에서 호스트 측 출력 스트림으로 형식화된 출력을 프린트합니다.
 
The in-kernel printf() function behaves in a similar way to the standard C-library printf() function, and the user is referred to the host system’s manual pages for a complete description of printf() behavior. 
> 커널 내 printf() 함수는 표준 C 라이브러리 printf() 함수와 비슷한 방식으로 작동하며 사용자는 printf() 동작에 대한 전체 설명을 보려면 호스트 시스템의 매뉴얼 페이지를 참조하십시오.

In essence, the string passed in as format is output to a stream on the host, with substitutions made from the argument list wherever a format specifier is encountered. Supported format specifiers are listed below. 
> 본질적으로 형식으로 전달된 문자열은 형식 지정자가 있는 곳의 변수 목록에서 대체하여 호스트의 스트림으로 출력됩니다. 지원되는 형식 지정자는 아래에 열거되어 있습니다.

The printf() command is executed as any other device-side function: per-thread, and in the context of the calling thread. 
> printf() 명령은 다른 모든 디바이스 측 함수처럼 실행됩니다 : 스레드마다, 그리고 호출 스레드의 컨텍스트에서.

From a multi-threaded kernel, this means that a straightforward call to printf() will be executed by every thread, using that thread’s data as specified. 
> 멀티 스레드 커널에서, 이것은 printf()에 대한 직접적인 호출이 모든 스레드에 의해 실행되고, 그 스레드의 데이터를 지정된 대로 사용한다는 것을 의미합니다.

Multiple versions of the output string will then appear at the host stream, once for each thread which encountered the printf(). 
> printf()가 발생한 각 스레드에 대해 출력 스트림의 여러 버전이 호스트 스트림에 나타납니다.

It is up to the programmer to limit the output to a single thread if only a single output string is desired (see Section B.16.4 for an illustrative example). 
> 프로그래머는 단일 출력 문자열만 원할 경우 출력을 단일 스레드로 제한해야 합니다 (예를 들어 설명하자면 B.16.4 절 참조).

Unlike the C-standard printf(), which returns the number of characters printed, CUDA’s printf() returns the number of arguments parsed. 
> 인쇄된 문자 수를 반환하는 C 표준 printf()와 달리 CUDA의 printf()는 구문 분석된 변수의 수를 반환합니다.

If no arguments follow the format string, 0 is returned. If the format string is NULL, -1 is returned. If an internal error occurs, -2 is returned.  
> 형식 문자열 뒤에 변수가 없으면 0이 반환됩니다. 형식 문자열이 NULL이면 -1이 반환됩니다. 내부 오류가 발생하면 -2가 반환됩니다.

## B.16.1 Format Specifiers 
> B.16.1 포맷 지정자

As for standard printf(), format specifiers take the form: 
> 표준 printf()에 관해서는, 포맷 지정자는 다음 형식을 취합니다. 

The following fields are supported (see widely-available documentation for a complete description of all behaviors):  
> 다음 필드가 지원됩니다 (모든 동작에 대한 전체 설명은 널리 사용 가능한 설명서 참조).

Note that CUDA’s printf() will accept any combination of flag, width, precision, size and type, whether or not overall they form a valid format specifier. 
> CUDA의 printf()는 플래그, 너비, 정밀도, 크기 및 유형이 유효한 포맷 지정자를 구성하는지 여부에 관계없이 모든 조합을 허용합니다.

In other words, “%hd” will be accepted and printf will expect a double-precision variable in the corresponding location in the argument list. 
> 즉, "% hd"가 허용되고 printf는 변수 목록의 해당 위치에 배정밀도 변수가 필요합니다.

## B.16.2 Limitations 
> B.16.2 제한 사항

Final formatting of the printf() output takes place on the host system. 
> printf() 출력의 최종 형식은 호스트 시스템에서 발생합니다.

This means that the format string must be understood by the host-system’s compiler and C library. 
> 즉, 형식 문자열은 호스트 시스템의 컴파일러 및 C 라이브러리에서 이해해야 합니다.

Every effort has been made to ensure that the format specifiers supported by CUDA’s printf function form a universal subset from the most common host compilers, but exact behavior will be host-O/S-dependent. 
> CUDA의 printf 함수가 지원하는 형식 지정자가 가장 일반적인 호스트 컴파일러에서 보편적 인 하위 집합을 형성하도록 모든 노력을 기울였지만 정확한 동작은 호스트 O/S 종속이 됩니다.

As described in Section B.16.1, printf() will accept all combinations of valid flags and types. 
> B.16.1 절에서 설명한 것처럼 printf()는 유효한 플래그와 유형의 모든 조합을 허용합니다.

This is because it cannot determine what will and will not be valid on the host system where the final output is formatted. 
> 이는 최종 출력이 형식화된 호스트 시스템에서 유효하고 유효하지 않은 것을 판별할 수 없기 때문입니다.

The effect of this is that output may be undefined if the program emits a format string which contains invalid combinations. 
> 이 효과는 프로그램이 잘못된 조합을 포함하는 형식 문자열을 방출하는 경우 출력이 정의되지 않을 수 있다는 것입니다.

The printf() command can accept at most 32 arguments in addition to the format string. 
> printf() 명령은 형식 문자열 외에도 최대 32 개의 변수를 채택할 수 있습니다.

Additional arguments beyond this will be ignored, and the format specifier output as-is. 
> 그 이상의 추가 변수는 무시되고 형식 지정자는 그대로 출력됩니다.

Owing to the differing size of the long type on 64-bit Windows platforms (four bytes on 64-bit Windows platforms, eight bytes on other 64-bit platforms), a kernel which is compiled on a non-Windows 64-bit machine but then run on a win64 machine will see corrupted output for all format strings which include “%ld”. 
> 64 비트 윈도우 플랫폼 (64 비트 윈도우 플랫폼에서는 4 바이트, 다른 64 비트 플랫폼에서는 8 바이트)에서 긴 유형의 크기가 다르기 때문에 비 윈도우 64 비트 시스템에서 컴파일되는 커널이지만 win64 시스템에서 실행하면 "% ld"가 포함된 모든 형식 문자열에 대해 손상된 출력을 보게 됩니다.

It is recommended that the compilation platform matches the execution platform to ensure safety. 
> 컴파일 플랫폼은 안전을 위해 실행 플랫폼과 일치하는 것이 좋습니다.

The output buffer for printf() is set to a fixed size before kernel launch (see Section B.16.3). 
> printf()의 출력 버퍼는 커널을 시작하기 전에 고정 크기로 설정됩니다 (B.16.3 절 참조).

It is circular and if more output is produced during kernel execution than can fit in the buffer, older output is overwritten. 
> 순환형이며 커널 실행 중에 버퍼에 저장할 수 있는 출력보다 많은 출력이 생성되면 이전 출력을 덮어씁니다.

It is flushed only when one of these actions is performed:  
> 다음 조치 중 하나가 수행될 때만 플러시됩니다.

Kernel launch via <<<>>> or cuLaunchKernel() (at the start of the launch, and if the CUDA_LAUNCH_BLOCKING environment variable is set to 1, at the end of the launch as well),   
> <<< >>> 또는 cuLaunchKernel()을 통해 커널을 시작합니다 (론칭을 시작할 때 CUDA_LAUNCH_BLOCKING 환경 변수가 1로 설정된 경우 론칭이 끝날 때까지). 

Memory copies via any blocking version of cudaMemcpy*() or cuMemcpy*(),  Module loading/unloading via cuModuleLoad() or cuModuleUnload(),  
> cudaMemcpy*() 또는 cuMemcpy*()의 차단 버전을 통한 메모리 사본, cuModuleLoad() 또는 cuModuleUnload()를 통한 모듈로드/언로드,

Context destruction via cudaDeviceReset() or cuCtxDestroy(). Note that the buffer is not flushed automatically when the program exits. 
> cudaDeviceReset() 또는 cuCtxDestroy()를 통한 컨텍스트 파기. 프로그램 종료시 버퍼가 자동으로 플러시되지 않습니다.

The user must call cudaDeviceReset() or cuCtxDestroy() explicitly, as shown in the examples below. 
> 사용자는 아래 예제에서와 같이 명시적으로 cudaDeviceReset() 또는 cuCtxDestroy()를 호출해야 합니다.

## B.16.3 Associated Host-Side API 
> B.16.3 관련 호스트 측 API

The following API functions get and set the size of the buffer used to transfer the printf() arguments and internal metadata to the host (default is 1 megabyte):
> 다음 API 함수는 printf() 변수와 내부 메타데이터를 호스트로 전송하는 데 사용되는 버퍼 크기를 가져오고 설정합니다 (기본값은 1MB).

Notice how each thread encounters the printf() command, so there are as many lines of output as there were threads launched in the grid. 
> 각 스레드가 printf() 명령을 만나면 그리드에서 론칭된 스레드 수 만큼 출력 라인이 표시됩니다.

As expected, global values (i.e. float f) are common between all threads, and local values (i.e. threadIdx.x) are distinct per-thread. The following code sample:    
> 예상대로 전역 값 (즉, float f)은 모든 스레드 간에 공통이고 로컬 값 (즉, threadIdx.x)은 스레드마다 고유합니다. 다음은 코드 샘플입니다.

the if() statement limits which threads will call printf, so that only a single line of output is seen.  
> if() 문은 printf를 호출할 스레드를 제한하여 출력의 한 줄만 볼 수 있도록 합니다.

## B.17 Dynamic Global Memory Allocation 
> B.17 동적 전역 메모리 할당

Dynamic global memory allocation is only supported by devices of compute capability 2.x and higher. 
> 동적 전역 메모리 할당은 컴퓨팅 기능이 2.x 이상인 디바이스에서만 지원됩니다.

void free(void* ptr); allocate and free memory dynamically from a fixed-size heap in global memory. 
> void free(void * ptr); 전역 메모리의 고정 크기 힙에서 동적으로 메모리를 할당하고 해제합니다.

The CUDA in-kernel malloc() function allocates at least size bytes from the device heap and returns a pointer to the allocated memory or NULL if insufficient memory exists to fulfill the request. 
> CUDA 커널 내 malloc() 함수는 최소한 디바이스 힙의 크기 바이트를 할당하고 할당된 메모리에 대한 포인터를 반환하거나 요청을 수행하기에 불충분한 메모리가 있는 경우 NULL을 반환합니다.

The returned pointer is guaranteed to be aligned to a 16-byte boundary. 
> 반환된 포인터는 16 바이트 경계로 정렬됩니다.

The CUDA in-kernel free() function deallocates the memory pointed to by ptr, which must have been returned by a previous call to malloc(). 
> CUDA 커널 내 free() 함수는 ptr이 가리키는 메모리를 할당 해제합니다. ptr은 이전에 malloc()을 호출하여 반환해야 합니다.

If ptr is NULL, the call to free() is ignored. Repeated calls to free() with the same ptr has undefined behavior. 
> ptr이 NULL이면 free()에 대한 호출이 무시됩니다. 동일한 ptr로 free()를 반복적으로 호출하면 정의되지 않은 동작이 발생합니다.

The memory allocated by a given CUDA thread via malloc() remains allocated for the lifetime of the CUDA context, or until it is explicitly released by a call to free(). 
> malloc()을 통해 지정된 CUDA 스레드에 의해 할당된 메모리는 CUDA 컨텍스트의 수명 동안 할당된 채로 남아 있거나 free()에 대한 호출에 의해 명시적으로 해제될 때까지 할당됩니다.

It can be used by any other CUDA threads even from subsequent kernel launches. 
> 후속 커널 론칭에서도 다른 CUDA 스레드에서 사용할 수 있습니다.

Any CUDA thread may free memory allocated by another thread, but care should be taken to ensure that the same pointer is not freed more than once. 
> 모든 CUDA 스레드는 다른 스레드가 할당한 메모리를 확보할 수 있지만 동일한 포인터가 두 번 이상 해제되지 않도록 주의해야 합니다.

## B.17.1 Heap Memory Allocation
> B.17.1 힙 메모리 할당
 
The device memory heap has a fixed size that must be specified before any program using malloc() or free() is loaded into the context. 
> 디바이스 메모리 힙은 malloc() 또는 free()를 사용하는 프로그램이 컨텍스트로 로드되기 전에 지정되어야 하는 고정 크기를 가집니다.

A default heap of eight megabytes is allocated if any program uses malloc() without explicitly specifying the heap size. The following API functions get and set the heap size:  
> 어떤 프로그램이 명시적으로 힙 크기를 지정하지 않고 malloc()을 사용하면 8 메가바이트의 기본 힙이 할당됩니다. 다음 API 함수는 힙 크기를 가져오고 설정합니다.

The heap size granted will be at least size bytes. cuCtxGetLimit() and cudaDeviceGetLimit() return the currently requested heap size. 
> 부여된 힙 크기는 최소한 크기 바이트가 됩니다. cuCtxGetLimit() 및 cudaDeviceGetLimit()은 현재 요청된 힙 크기를 반환합니다.

The actual memory allocation for the heap occurs when a module is loaded into the context, either explicitly via the CUDA driver API (see Section G.2), or implicitly via the CUDA runtime API (see Section 3.2). 
> 힙에 대한 실제 메모리 할당은 모듈이 CUDA 드라이버 API (G.2 절 참조) 또는 CUDA 런타임 API (3.2 절 참조)를 통해 명시적으로 컨텍스트에 로드될 때 발생합니다.

If the memory allocation fails, the module load will generate a CUDA_ERROR_SHARED_OBJECT_INIT_FAILED error. 
> 메모리 할당이 실패하면 모듈 로드가 CUDA_ERROR_SHARED_OBJECT_INIT_FAILED 오류를 생성합니다.

Heap size cannot be changed once a module load has occurred and it does not resize dynamically according to need. 
> 모듈 로드가 발생하고 필요에 따라 동적으로 크기가 조정되지 않으면 힙 크기를 변경할 수 없습니다.

Memory reserved for the device heap is in addition to memory allocated through host-side CUDA API calls such as cudaMalloc(). 
> 디바이스 힙용으로 예약된 메모리는 cudaMalloc()과 같은 호스트 측 CUDA API 호출을 통해 할당된 메모리에 추가됩니다.
 
## B.17.2 Interoperability with Host Memory API
> B.17.2 호스트 메모리 API와의 상호운용
 
Memory allocated via malloc() cannot be freed using the runtime (i.e. by calling any of the free memory functions from Sections 3.2.2). 
> malloc()을 통해 할당된 메모리는 런타임을 사용하여 (즉 3.2.2 절의 사용 가능한 메모리 함수 중 하나를 호출하여) 해제할 수 없습니다.

Similarly, memory allocated via the runtime (i.e. by calling any of the memory allocation functions from Sections 3.2.2) cannot be freed via free(). 
> 마찬가지로 런타임을 통해 할당된 메모리 (즉, 3.2.2 절의 메모리 할당 함수 중 하나를 호출하여)는 free()를 통해 해제할 수 없습니다.

Memory allocated via malloc() can be copied using the runtime (i.e. by calling any of the copy memory functions from Sections 3.2.2). 
> malloc()을 통해 할당된 메모리는 런타임을 사용하여 복사할 수 있습니다 (즉, 3.2.2 절의 복사 메모리 함수 중 하나를 호출하여 복사할 수 있음).

## B.17.3 Examples 
> B.17.3 예제

## B.17.3.1 Per Thread Allocation 
> B.17.3.1 스레드 별 할당 

The following code sample:
> 코드 샘플은 다음과 같습니다.

Notice how each thread encounters the malloc() command and so receives its own allocation. (Exact pointer values will vary: these are illustrative.) 
> 각 스레드가 malloc() 명령을 만나고 자체 할당을 수신하는 방법에 주목하십시오. (정확한 포인터 값은 다양합니다: 이것들은 예시입니다.) 

## B.17.3.2 Per Thread Block Allocation 
> B.17.3.2 스레드 별 블록 할당

Only the first thread in the block does the allocation     // since we want only one allocation per block.    
> 블록 당 하나의 할당만 필요하기 때문에 블록의 첫 번째 스레드만 할당을 수행합니다.

## B.18 Execution Configuration 
> B.18 실행 구성

Any call to a __global__ function must specify the execution configuration for that call. 
> __global__ 함수에 대한 호출은 해당 호출의 실행 구성을 지정해야 합니다.

The execution configuration defines the dimension of the grid and blocks that will be used to execute the function on the device, as well as the associated stream (see Section 3.2.5.5 for a description of streams). 
> 실행 설정은 디바이스의 기능을 실행하는 데 사용되는 그리드와 블록의 차원 및 관련 스트림을 정의합니다 (스트림 설명은 3.2.5.5 절 참조).

The execution configuration is specified by inserting an expression of the form <<< Dg, Db, Ns, S >>> between the function name and the parenthesized argument list, where:  
> 실행 구성은 함수 이름과 괄호 안의 변수 목록 사이에 <<< Dg, Db, Ns, S >>> 형식의 식을 삽입하여 지정합니다. 

Dg is of type dim3 (see Section B.3.2) and specifies the dimension and size of the grid, such that Dg.x * Dg.y * Dg.z equals the number of blocks being launched; Dg.z must be equal to 1 for devices of compute capability 1.x; 
>  Dg는 dim3 유형이며 (B.3.2 절 참조) 그리드의 차원과 크기를 지정하여 Dg.x * Dg.y * Dg.z가 시작되는 블록의 수와 같습니다. 컴퓨팅 기능 1.x의 디바이스의 경우 Dg.z는 1과 같아야 합니다.

Db is of type dim3 (see Section B.3.2) and specifies the dimension and size of each block, such that Db.x * Db.y * Db.z equals the number of threads per block;  
> Db는 dim3 유형이며 (B.3.2 절 참조) Db.x * Db.y * Db.z가 블록 당 스레드 수와 같도록 각 블록의 차원과 크기를 지정합니다.

Ns is of type size_t and specifies the number of bytes in shared memory that is dynamically allocated per block for this call in addition to the statically allocated memory;
> Ns는 size_t 유형이며 공유 메모리에 정적으로 할당된 메모리 외에도 이 호출에 대해 블록 당 동적으로 할당되는 바이트 수를 지정합니다.

this dynamically allocated memory is used by any of the variables declared as an external array as mentioned in Section B.2.3; 
> 이 동적으로 할당된 메모리는 B.2.3 절에서 언급한 것처럼 외부 배열로 선언된 변수 중 하나에서 사용됩니다.

Ns is an optional argument which defaults to 0;  S is of type cudaStream_t and specifies the associated stream; S is an optional argument which defaults to 0. 
> Ns는 선택적 변수이며 기본값은 0입니다. S는 cudaStream_t 유형이며 연관된 스트림을 지정합니다. S는 선택적 변수이며 기본값은 0입니다.

As an example, a function declared as __global__ void Func(float* parameter); must be called like this: 
> 예를 들어, __global__으로 선언된 함수 void Func (float* parameter); 다음과 같이 호출해야 합니다 : 

The arguments to the execution configuration are evaluated before the actual function arguments and like the function arguments, are currently passed via shared memory to the device. 
> 실행 구성에 대한 변수는 실제 함수 변수와 함수 변수와 같은 현재 공유 메모리를 통해 디바이스로 전달됩니다.

The function call will fail if Dg or Db are greater than the maximum sizes allowed for the device as specified in Appendix F, or if Ns is greater than the maximum amount of shared memory available on the device, minus the amount of shared memory required for static allocation, functions arguments (for devices of compute capability 1.x), and execution configuration.  
> Dg 또는 Db가 부록 F에 지정된 대로 디바이스에 허용된 최대 크기보다 큰 경우, 또는 Ns가 디바이스에서 사용 가능한 공유 메모리의 최대 크기보다 큰 경우, 정적 할당에 필요한 공유 메모리 양, 함수 변수 (컴퓨팅 기능 1.x의 디바이스 용) 및 실행 구성을 뺀 경우 함수 호출이 실패합니다.

## B.19 Launch Bounds 
> B.19 론칭 범위

As discussed in detail in Section 5.2.3, the fewer registers a kernel uses, the more threads and thread blocks are likely to reside on a multiprocessor, which can improve performance.
> 5.2.3 절에서 자세히 설명했듯이, 커널이 사용하는 레지스터 수가 적을수록 다중프로세서에 더 많은 스레드와 스레드 블록이 상주하여 성능을 향상시킬 수 있습니다.

Therefore, the compiler uses heuristics to minimize register usage while keeping register spilling (see Section 5.3.2.2) and instruction count to a minimum. 
> 따라서 컴파일러는 휴리스틱을 사용하여 레지스터 유출 (5.3.2.2 절 참조) 및 명령 수를 최소로 유지하면서 레지스터 사용을 최소화합니다.

An application can optionally aid these heuristics by providing additional information to the compiler in the form of launch bounds that are specified using the __launch_bounds__() qualifier in the definition of a __global__ function: 
> 애플리케이션은 선택적으로 __global__ 함수의 정의에서 __launch_bounds__() 한정자를 사용하여 지정되는 론칭 경계 형태로 컴파일러에 추가 정보를 제공하여 이러한 휴리스틱(추론)을 지원할 수 있습니다.

maxThreadsPerBlock specifies the maximum number of threads per block with which the application will ever launch MyKernel(); it compiles to the .maxntid PTX directive;  
> maxThreadsPerBlock은 애플리케이션이 MyKernel()을 시작할 블록 당 최대 스레드 수를 지정합니다. 이것은 .maxntid PTX 지시문으로 컴파일됩니다.

minBlocksPerMultiprocessor is optional and specifies the desired minimum number of resident blocks per multiprocessor; it compiles to the .minnctapersm PTX directive. 
> minBlocksPerMultiprocessor는 선택적이며 다중프로세서 당 원하는 상주 블록 수를 지정합니다. 그것은 .minnctapersm PTX 지시어로 컴파일됩니다.
 
If launch bounds are specified, the compiler first derives from them the upper limit L on the number of registers the kernel should use to ensure that minBlocksPerMultiprocessor blocks (or a single block if minBlocksPerMultiprocessor is not specified) of maxThreadsPerBlock threads can reside on the multiprocessor (see Section 4.2 for the relationship between the number of registers used by a kernel and the number of registers allocated per block). 
> 시작 경계가 지정되면 컴파일러는 maxThreadsPerBlock 스레드의 minBlocksPerMultiprocessor 블록 (또는 minBlocksPerMultiprocessor가 지정되지 않은 경우 단일 블록)이 다중프로세서에 상주할 수 있도록 커널이 사용해야 하는 레지스터 수의 상한값 L을 먼저 가져옵니다 ( 커널이 사용하는 레지스터의 수와 블록 당 할당된 레지스터의 수 사이의 관계는 4.2 절을 참조하십시오).

The compiler then optimizes register usage in the following way:  
> 그런 다음 컴파일러는 다음과 같은 방법으로 레지스터 사용을 최적화합니다.

If the initial register usage is higher than L, the compiler reduces it further until it becomes less or equal to L, usually at the expense of more local memory usage and/or higher number of instructions;  
> 초기 레지스터 사용량이 L보다 높으면 컴파일러는 L보다 작거나 같아질 때까지 이를 더 줄입니다. 일반적으로 로컬 메모리 사용량 및/또는 명령어 수가 더 많습니다.

If the initial register usage is lower than L,  
> 초기 레지스터 사용량이 L보다 작은 경우,

If maxThreadsPerBlock is specified and minBlocksPerMultiprocessor is not, the compiler uses maxThreadsPerBlock to determine the register usage thresholds for the transitions between n and n+1 resident blocks (i.e. when using one less register makes room for an additional resident block as in the example of Section 5.2.3) and then applies similar heuristics as when no launch bounds are specified;  
> maxThreadsPerBlock이 지정되고 minBlocksPerMultiprocessor가 지정되지 않은 경우, 컴파일러는 maxThreadsPerBlock을 사용하여 n 및 n+1 상주 블록 사이의 전환에 대한 레지스터 사용 임계값을 결정합니다 (즉, 하나의 적은 레지스터를 사용하면 5.2.3 절의 예와 같이 추가 상주 블록을 위한 공간을 만듭니다). 그리고 론칭 범위가 지정되지 않은 경우와 유사한 휴리스틱을 적용합니다.

If both minBlocksPerMultiprocessor and maxThreadsPerBlock are specified, the compiler may increase register usage as high as L to reduce the number of instructions and better hide single thread instruction latency. 
> minBlocksPerMultiprocessor와 maxThreadsPerBlock이 둘 다 지정되면 컴파일러는 명령어 수를 줄이고 단일 스레드 명령어 대기 시간을 보다 잘 숨기기 위해 레지스터 사용을 L만큼 증가시킬 수 있습니다.

A kernel will fail to launch if it is executed with more threads per block than its launch bound maxThreadsPerBlock. 
> 론칭 경계 maxThreadsPerBlock보다 블록 당 더 많은 스레드를 사용하여 커널을 실행하면 커널이 론칭되지 않습니다.

Optimal launch bounds for a given kernel will usually differ across major architecture revisions. 
> 지정된 커널에 대한 최적의 론칭 경계는 대개 주요 아키텍처 수정에 따라 다릅니다.

The sample code below shows how this is typically handled in device code using the __CUDA_ARCH__ macro introduced in Section 3.1.4.
> 아래의 샘플 코드는 3.1.4 절에서 소개한 __CUDA_ARCH__ 매크로를 사용하여 디바이스 코드에서 일반적으로 어떻게 처리되는지 보여줍니다.

In the common case where MyKernel is invoked with the maximum number of threads per block (specified as the first parameter of __launch_bounds__()), 
> 일반적으로 MyKernel이 블록 당 최대 스레드 수 (__launch_bounds __()의 첫 번째 매개변수로 지정됨)로 호출되는 경우,

it is tempting to use MY_KERNEL_MAX_THREADS as the number of threads per block in the execution configuration:
> 실행 구성에서 블록 당 스레드 수로 MY_KERNEL_MAX_THREADS를 사용하려고 합니다.
 
This will not work however since __CUDA_ARCH__ is undefined in host code as mentioned in Section 3.1.4, so MyKernel will launch with 256 threads per block even when __CUDA_ARCH__ is greater or equal to 200. 
> 그러나 __CUDA_ARCH__가 3.1.4 절에서 언급한 것처럼 호스트 코드에서 정의되지 않았으므로 MyKernel은 __CUDA_ARCH__가 200보다 크거나 같은 경우에도 블록 당 256 개의 스레드로 실행됩니다. 

Instead the number of threads per block should be determined:
> 대신 블록 당 스레드 수를 결정해야 합니다.

Either at compile time using a macro that does not depend on __CUDA_ARCH__, for example // 
> __CUDA_ARCH__에 의존하지 않는 매크로를 사용해서 어느 쪽이든 컴파일할 때 (예를 들어 : //).

Host code MyKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(...);  Or at runtime based on the compute capability 
> 호스트 코드 MyKernel <<< blocksPerGrid, THREADS_PER_BLOCK >>> (...); 또는 컴퓨팅 기능을 기반으로 런타임에

Register usage is reported by the --ptxas-options=-v compiler option. 
> 레지스터 사용은 --ptxas-options=-v 컴파일러 옵션에 의해 보고됩니다.

The number of resident blocks can be derived from the occupancy reported by the CUDA profiler (see Section 5.2.3 for a definition of occupancy). 
> 상주 블록의 수는 CUDA 프로파일러에 의해 보고된 점유에서 파생될 수 있습니다 (점유의 정의에 대해서는 5.2.3 절 참조).

Register usage can also be controlled for all __global__ functions in a file using the -maxrregcount compiler option. 
> 레지스터 사용은 -maxrregcount 컴파일러 옵션을 사용하여 파일의 모든 __global__ 함수에 대해서도 제어할 수 있습니다.

The value of -maxrregcount is ignored for functions with launch bounds. 
> 론칭 경계가 있는 함수의 경우 -maxrregcount 값이 무시됩니다.

## B.20 #pragma unroll 
> B.20 #pragma unroll 

By default, the compiler unrolls small loops with a known trip count. 
> 기본적으로 컴파일러는 알려진 트립 카운트의 작은 루프를 전개합니다.

The #pragma unroll directive however can be used to control unrolling of any given loop. 
> 그러나 #pragma unroll 지시어는 지정된 루프의 언롤링(롤 풀기)을 제어하는 데 사용할 수 있습니다.

It must be placed immediately before the loop and only applies to that loop. 
> 루프 바로 앞에 배치해야 하며 해당 루프에만 적용됩니다.

It is optionally followed by a number that specifies how many times the loop must be unrolled. 
> 선택적으로 루프가 풀려야 하는 횟수를 지정하는 숫자가 뒤에 옵니다.

For example, in this code sample: #pragma unroll 5 for (int i = 0; i < n; ++i) the loop will be unrolled 5 times. 
> 예를 들어 이 코드 샘플에서 #pragma unroll 5 for (int i = 0; i <n; ++ i)는 루프가 5 번 풀립니다.

The compiler will also insert code to ensure correctness (in the example above, to ensure that there will only be n iterations if n is less than 5, for example). 
> 컴파일러는 정확성을 보장하는 코드를 삽입합니다 (위의 예에서, n이 5 미만인 경우 n 번만 반복하도록 합니다 ).

It is up to the programmer to make sure that the specified unroll number gives the best performance. 
> 지정된 언롤 (unroll) 번호가 최상의 성능을 제공하는지 확인하는 것은 프로그래머의 몫입니다. 

#pragma unroll 1 will prevent the compiler from ever unrolling a loop.
> #pragma unroll 1은 컴파일러가 루프를 언롤링(푸는)하는 것을 방지합니다.

If no number is specified after #pragma unroll, the loop is completely unrolled if its trip count is constant, otherwise it is not unrolled at all. 
> #pragma unroll 이후에 번호가 지정되지 않은 경우 반복 횟수가 일정하면 루프가 완전히 풀립니다. 그렇지 않으면 루프가 전혀 실행되지 않습니다.
