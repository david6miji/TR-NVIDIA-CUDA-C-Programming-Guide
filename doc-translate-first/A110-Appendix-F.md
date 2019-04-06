## Appendix F. Compute Capabilities 
> 부록 F. 컴퓨팅 기능

The general specifications and features of a compute device depend on its compute capability (see Section 2.5). 
> 컴퓨팅 디바이스의 일반적인 사양과 기능은 컴퓨팅 기능에 따라 다릅니다 (2.5 절 참조).

Section F.1 gives the features and technical specifications associated to each compute capability. 
> F.1 절은 각 컴퓨팅 기능과 관련된 기능 및 기술 사양을 제공합니다.

Section F.2 reviews the compliance with the IEEE floating-point standard. 
> F.2 절에서는 IEEE 부동 소수점 표준 준수 여부를 검토합니다.

Section F.3, F.4, and F.5 give more details on the architecture of devices of compute capability 1.x, 2.x, and 3.0, respectively.   
> F.3, F.4 및 F.5 절은 각각 컴퓨팅 기능 1.x, 2.x 및 3.0의 디바이스 아키텍처에 대한 자세한 내용을 제공합니다.
 
F.1 Features and Technical Specifications 
> F.1 기능 및 기술 사양

Table F-1. Feature Support per Compute Capability 
> 표 F-1. 컴퓨팅 기능별 기능 지원

Compute Capability Feature Support (Unlisted features are supported for all compute capabilities) 
> 컴퓨팅 기능 기능 지원 (모든 컴퓨팅 기능에 대해 나열되지 않은 기능이 지원됩니다)

Atomic functions operating on 32-bit integer values in global memory (Section B.11) 
> 전역 메모리의 32 비트 정수 값에서 작동하는 원자 함수 (B.11 절)

No Yes 
> 아니요 예

atomicExch() operating on 32-bit floating point values in global memory (Section B.11.1.3) 
> 전역 메모리의 32 비트 부동 소수점 값에서 작동하는 atomicExch() (B.11.1.3 절)

Atomic functions operating on 32-bit integer values in shared memory (Section B.11) 
> 공유 메모리의 32 비트 정수 값에서 작동하는 원자 함수 (B.11 절)

No Yes 
> 아니요 예

atomicExch() operating on 32-bit floating point values in shared memory (Section B.11.1.3) 
> 공유 메모리의 32 비트 부동 소수점 값에서 작동하는 atomicExch() (B.11.1.3 절).

Atomic functions operating on 64-bit integer values in global memory (Section B.11) 
> 전역 메모리의 64 비트 정수 값에서 작동하는 원자 함수 (B.11 절)

Warp vote functions (Section B.12) 
> 워프 투표 함수 (B.12 절)

Double-precision floating-point numbers 
> 배 정밀도 부동 소수점 수

No Yes 
> 아니요 예

Atomic functions operating on 64-bit integer values in shared memory (Section B.11) 
> 공유 메모리의 64 비트 정수 값에서 작동하는 원자 함수 (B.11 절)

No Yes 
> 아니요 예

Atomic addition operating on 32-bit floating point values in global and shared memory (Section B.11.1.1) 
> 전역 및 공유 메모리 (B.11.1.1 절)의 32 비트 부동 소수점 값에서 작동하는 원자 추가

__ballot() (Section B.12) __threadfence_system() (Section B.5) __syncthreads_count(), __syncthreads_and(), __syncthreads_or() (Section B.6) Surface functions (Section B.9) 3D grid of thread blocks 
 
Table F-2. Technical Specifications per Compute Capability  
> 표 F-2. 컴퓨팅 기능 당 기술 사양

Compute Capability Technical Specifications 1.0 1.1 1.2 1.3 2.x 3.0 
> 컴퓨팅 기능 기술 사양 1.0 1.1 1.2 1.3 2.x 3.0

Maximum dimensionality of grid of thread blocks 2 3 
> 스레드 블록 2 3의 그리드의 최대 치수(차원)

Maximum x-dimension of a grid of thread blocks 65535 231-1 Maximum y- or z- 65535 
> 스레드 블록 그리드의 최대 x 차원(크기) 65535 231-1 최대 y 또는 z 65535
 
Compute Capability Technical Specifications 1.0 1.1 1.2 1.3 2.x 3.0 dimension of a grid of thread blocks 
> 컴퓨팅 기능 기술 사양 1.0 1.1 1.2 1.3 2.x 3.0 스레드 블록 그리드의 차원(크기)

Maximum dimensionality of thread block 3 
> 스레드 블록 3의 최대 차원

Maximum x- or ydimension of a block 
> 블록 512 1024의 최대 x- 또는 y- 차원 512 1024

Maximum z-dimension of a block 64 
> 블록 64의 최대 z- 차원

Maximum number of threads per block 512 1024 
> 블록 당 최대 스레드 수 512 1024

Warp size 32 Maximum number of resident blocks per multiprocessor 8 16 
> 워프 크기 32 다중프로세서 당 최대 상주 블록 수 8 16 

Maximum number of resident warps per multiprocessor 24 32 48 64 
> 다중프로세서 당 최대 상주 워프 수 24 32 48 64

Maximum number of resident threads per multiprocessor 768 1024 1536 2048 
> 다중프로세서 당 최대 상주 스레드 수 768 1024 1536 2048

Number of 32-bit registers per multiprocessor 8 K 16 K 32 K 64 K 
> 다중프로세서 당 32 비트 레지스터 수 8 K 16 K 32 K 64 K

Maximum amount of shared memory per multiprocessor 16 KB 48 KB 
> 다중프로세서 당 최대 공유 메모리 크기 16KB 48KB

Number of shared memory banks 16 32 
> 공유 메모리 뱅크 수 16 32

Amount of local memory per thread 16 KB 512 KB 
> 스레드 당 로컬 메모리 양 16 KB 512 KB

Constant memory size 64 KB 
> 상수 메모리 크기 64KB

Cache working set per multiprocessor for constant memory 8 KB 
> 상수 메모리를 위한 다중프로세서 당 캐시 작업 세트 8KB

Cache working set per multiprocessor for texture memory 
> 텍스처 메모리를 위한 다중프로세서 당 캐시 작업 세트

Device dependent, between 6 KB and 8 KB Maximum width for a 1D texture reference bound to a CUDA array 8192 65536 
> 디바이스 의존, 6KB에서 8KB 사이 CUDA 배열에 바인딩된 1D 텍스처 참조의 최대 너비 8192 65536

Maximum width for a 1D texture reference bound to linear memory 227
> 선형 메모리에 바인딩된 1D 텍스처 참조의 최대 너비 227

Maximum width and number of layers for a 1D layered texture reference  8192 x 512 16384 x 2048 
> 1D 계층화된 텍스처 참조에 대한 최대 너비와 계층 수 8192 x 512 16384 x 2048

Maximum width and height for a 2D texture reference bound to a CUDA array 65536 x 32768 65536 x 65535 
> CUDA 배열에 바인딩된 2D 텍스처 참조의 최대 너비와 높이 65536 x 32768 65536 x 65535
 
Compute Capability Technical Specifications 1.0 1.1 1.2 1.3 2.x 3.0 
> 컴퓨팅 기능 기술 사양 1.0 1.1 1.2 1.3 2.x 3.0

Maximum width and height for a 2D texture reference bound to linear memory 65000 x 65000 65000 x 65000 
> 선형 메모리에 바인딩된 2D 텍스처 참조의 최대 너비와 높이 65000 x 65000 65000 x 65000

Maximum width and height for a 2D texture reference bound to a CUDA array supporting texture gather N/A 16384 x 16384 
> 텍스처 수집을 지원하는 CUDA 배열에 바인딩된 2D 텍스처 참조의 최대 너비와 높이 N / A 16384 x 16384

Maximum width, height, and number of layers for a 2D layered texture reference 8192 x 8192 x 512 16384 x 16384 x 2048 
> 2D 계층화된 텍스처 참조의 최대 너비, 높이 및 계층 수 8192 x 8192 x 512 16384 x 16384 x 2048

Maximum width, height, and depth for a 3D texture reference bound to a CUDA array 2048 x 2048 x 2048 4096 x 4096 x 4096 
> CUDA 배열에 바인딩된 3D 텍스처 참조의 최대 너비, 높이 및 깊이 2048 x 2048 x 2048 4096 x 4096 x 4096

Maximum width (and height) for a cubemap texture reference N/A 16384 
> 큐브맵 텍스처 참조의 최대 너비 (및 높이) 해당 사항 없음 16384

Maximum width (and height) and number of layers for a cubemap layered texture reference N/A 16384 x 2046 
> 큐브맵 계층화된 텍스처 참조의 최대 너비 (및 높이)와 계층 수 N / A 16384 x 2046 

Maximum number of textures that can be bound to a kernel 128 256
> 커널에 바인딩할 수 있는 텍스처의 최대 수 128 256

Maximum width for a 1D surface reference bound to a CUDA array 
N/A 65536 
> CUDA 배열에 바인딩된 1D 표면 참조의 최대 너비

Maximum width and number of layers for a 1D layered surface reference 65536 x 2048 
> 1D 계층화된 표면 참조의 최대 너비 및 수 65536 x 2048

Maximum width and height for a 2D surface reference bound to a CUDA array 65536 x 32768 
> CUDA 배열에 바인딩된 2D 표면 참조의 최대 너비와 높이 65536 x 32768

Maximum width, height, and number of layers for a 2D layered surface reference 65536 x 32768 x  2048 
> 2D 계층화된 표면 참조의 최대 너비, 높이 및 계층 수 65536 x 32768 x 2048

Maximum width, height, and depth for a 3D surface reference bound to a CUDA array 
65536 x 32768 x 2048 
> CUDA 배열에 바인딩된 3D 표면 참조의 최대 너비, 높이 및 깊이 65536 x 32768 x 2048

Maximum width (and height) for a cubemap surface reference bound to a CUDA array 
32768 
> CUDA 배열에 바인딩된 큐브맵 표면 참조의 최대 너비 (및 높이) 32768

Maximum width (and height) and number of layers for a cubemap layered surface reference 
32768 x 2046 
> 큐브맵 계층화된 표면 참조의 최대 너비 (및 높이)와 계층 수 32768 x 2046
 
Compute Capability Technical Specifications 1.0 1.1 1.2 1.3 2.x 3.0 
> 컴퓨팅 기능 기술 사양 1.0 1.1 1.2 1.3 2.x 3.0

Maximum number of surfaces that can be bound to a kernel 8 16 
> 커널에 바인딩할 수있는 최대 표면 수 8 16

Maximum number of instructions per kernel 2 million 512 million 
> 커널 당 최대 명령 수 2 백만 5 억 2 천만

## F.2 Floating-Point Standard 
> F.2 부동 소수점 표준

All compute devices follow the IEEE 754-2008 standard for binary floating-point arithmetic with the following deviations:  
> 모든 컴퓨팅 디바이스는 IEEE 754-2008 표준을 준수하여 다음과 같은 편차를 갖는 이진 부동 소수점 연산을 지원합니다.

There is no dynamically configurable rounding mode; however, most of the operations support multiple IEEE rounding modes, exposed via device intrinsics;  
> 동적으로 구성 가능한 반올림 모드는 없습니다. 하지만 대부분의 연산은 디바이스 내장 함수를 통해 노출되는 여러 IEEE 반올림 모드를 지원합니다.

There is no mechanism for detecting that a floating-point exception has occurred and all operations behave as if the IEEE-754 exceptions are always masked, and deliver the masked response as defined by IEEE-754 if there is an exceptional event; 
> IEEE-754 예외가 항상 마스킹된(숨은) 것처럼 부동 소수점 예외가 발생하고 모든 연산이 동작하는 것으로 감지하는 메커니즘이 없으며 예외적인 이벤트가 있는 경우 IEEE-754에서 정의된 마스킹된 응답을 전달합니다.

for the same reason, while SNaN encodings are supported, they are not signaling and are handled as quiet;  
> 같은 이유로 SNaN 인코딩은 지원되지만 시그널링이 아니며 조용하게 처리됩니다.

The result of a single-precision floating-point operation involving one or more input NaNs is the quiet NaN of bit pattern 0x7fffffff;  
> 하나 이상의 입력 NaN이 포함된 단 정밀도 부동 소수점 연산의 결과는 비트 패턴 0x7fffffff의 조용한 NaN입니다.

Double-precision floating-point absolute value and negation are not compliant with IEEE-754 with respect to NaNs; these are passed through unchanged;  
> 배 정밀도 부동 소수점 절대값 및 부정은 NaN과 관련하여 IEEE-754와 호환되지 않습니다. 이들은 변경 없이 전달됩니다.

For single-precision floating-point numbers on devices of compute capability 1.x:  
> 컴퓨팅 기능 1.x의 디바이스인 단 정밀도 부동 소수점 숫자의 경우 :

Denormalized numbers are not supported; floating-point arithmetic and comparison instructions convert denormalized operands to zero prior to the floating-point operation;  
> 비정규 숫자는 지원되지 않습니다. 부동 소수점 산술 및 비교 명령어는 부동 소수점 연산 이전에 비정규 피연산자를 0으로 변환합니다.

Underflowed results are flushed to zero;  Some instructions are not IEEE-compliant:  
> 언더플로우된(저류인) 결과는 0으로 플러시됩니다(내보냅니다). 일부 명령은 IEEE와 호환되지 않습니다.

Addition and multiplication are often combined into a single multiply-add instruction (FMAD), which truncates (i.e. without rounding) the intermediate mantissa of the multiplication;  
> 가산 및 곱셈은 종종 곱셈의 중간 가수를 절단 (즉, 반올림하지 않고)하는 단일 다중 곱셈 덧셈 명령 (FMAD)으로 결합됩니다;

Division is implemented via the reciprocal in a non-standard-compliant way;  
> 나눗셈은 비표준 방식으로 상호를 통해 구현됩니다.

Square root is implemented via the reciprocal square root in a nonstandard-compliant way;  
> 제곱근은 비표준 방식으로 역수 제곱근을 통해 구현됩니다.

For addition and multiplication, only round-to-nearest-even and round-towards-zero are supported via static rounding modes; directed rounding towards +/- infinity is not supported; 
> 더하기 및 곱셈의 경우, 짝수에 가장 가까운 반올림 및 0으로 반올림은 정적 반올림 모드를 통해 지원됩니다.  +/- 무한대 방향의 반올림은 지원되지 않습니다.

To mitigate the impact of these restrictions, IEEE-compliant software (and therefore slower) implementations are provided through the following intrinsics (c.f. Section C.2.1): 
> 이러한 제한 사항의 영향을 완화하기 위해 IEEE 호환 소프트웨어 (따라서 느린) 구현은 다음 내장 함수를 통해 제공됩니다 (C.2.1 절 참조).

__fmaf_r{n,z,u,d}(float, float, float): single-precision fused multiply-add with IEEE rounding modes,  
> __fmaf_r {n, z, u, d} (float, float, float): IEEE 반올림 모드가 있는 단 정밀도 융합된 곱셈 덧셈,

__frcp_r[n,z,u,d](float): single-precision reciprocal with IEEE rounding modes,
> __frcp_r [n, z, u, d] (float): IEEE 반올림 모드의 단 정밀도 역수,

__fdiv_r[n,z,u,d](float, float): single-precision division with IEEE rounding modes,  
> __fdiv_r [n, z, u, d] (float, float): IEEE 반올림 모드가 있는 단 정밀도 나누기,

__fsqrt_r[n,z,u,d](float): single-precision square root with IEEE rounding modes,  
> __fsqrt_r [n, z, u, d] (float): IEEE 반올림 모드가 있는 단 정밀도 제곱근,

__fadd_r[u,d](float, float): single-precision addition with IEEE directed rounding,  
> __fadd_r [u, d] (float, float): IEEE 방향 반올림을 사용한 단 정밀도 추가,

__fmul_r[u,d](float, float): single-precision multiplication with IEEE directed rounding;  
> __fmul_r [u, d] (float, float): IEEE 방향 반올림을 사용하는 단 정밀도 곱셈.

For double-precision floating-point numbers on devices of compute capability 1.x:  
> 컴퓨팅 기능 1.x의 배 정밀도 부동 소수점 숫자의 경우 :

Round-to-nearest-even is the only supported IEEE rounding mode for reciprocal, division, and square root. 
> Round-to-nearest-even(짝수에 가장 가까운 반올림)은 역수, 나누기 및 제곱근에 대해 지원되는 유일한 IEEE 반올림 모드입니다.

When compiling for devices without native double-precision floating-point support, i.e. devices of compute capability 1.2 and lower, each double variable is converted to single-precision floating-point format (but retains its size of 64 bits) and double-precision floating-point arithmetic gets demoted to single-precision floating-point arithmetic. 
> 고유한 배 정밀도 부동 소수점 지원이 없는 디바이스, 즉 컴퓨팅 기능이 1.2 이하인 디바이스를 컴파일할 때, 각 이중 변수는 단 정밀도 부동 소수점 형식으로 변환되지만 크기는 64 비트로 유지됩니다. 배 정밀도 부동 소수점 산술은 단 정밀도 부동 소수점 산술로 강등됩니다.

For devices of compute capability 2.x and higher, code must be compiled with -ftz=false, -prec-div=true, and -prec-sqrt=true to ensure IEEE compliance (this is the default setting; see the nvcc user manual for description of these compilation flags);
> 컴퓨팅 성능이 2.x 이상인 디바이스의 경우, IEEE 준수 여부를 확인하려면 코드를 -ftz=false, -prec-div=true 및 -prec-sqrt=true로 컴파일해야 합니다 (이는 기본 설정입니다. 이러한 컴파일 플래그에 대한 설명은 nvcc 사용자 설명서를 참조하십시오). 

code compiled with -ftz=true, -prec-div=false, and -prec-sqrt=false comes closest to the code generated for devices of compute capability 1.x. 
> -ftz=true, -prec-div=false 및 -prec-sqrt=false로 컴파일된 코드는 컴퓨팅 기능 1.x의 디바이스으로 생성된 코드와 가장 유사합니다.

Addition and multiplication are often combined into a single multiply-add instruction:  
> 덧셈과 곱셈은 종종 하나의 곱셈-덧셈 명령으로 결합됩니다.

FMAD for single precision on devices of compute capability 1.x,  FFMA for single precision on devices of compute capability 2.x and higher. 
> 컴퓨팅 기능 1.x의 디바이스에서 단 정밀도의 FMAD, 컴퓨팅 기능 2.x 이상의 디바이스에서 단 정밀도의 FFMA.

As mentioned above, FMAD truncates the mantissa prior to use it in the addition. 
> 위에서 언급했듯이 FMAD는 덧셈으로 사용하기 전에 가수를 잘라냅니다.

FFMA, on the other hand, is an IEEE-754(2008) compliant fused multiply-add instruction, so the full-width product is being used in the addition and a single rounding occurs during generation of the final result. 
> 반면에 FFMA는 IEEE-754(2008) 호환 퓨즈된 곱셈 덧셈 명령어이므로 전체 폭 곱을 더하기에 사용되고 최종 결과 생성 중에 단일 반올림이 발생합니다.

While FFMA in general has superior numerical properties compared to FMAD, the switch from FMAD to FFMA can cause slight changes in numeric results and can in rare circumstances lead to slighty larger error in final results. 
> 일반적으로 FFMA는 FMAD에 비해 우수한 수치적 특성을 가지고 있지만 FMAD에서 FFMA로 전환하면 수치 결과가 약간 변경될 수 있으며 드문 경우이지만 최종 결과에서 약간의 큰 오류가 발생할 수 있습니다.

In accordance to the IEEE-754R standard, if one of the input parameters to fminf(), fmin(), fmaxf(), or fmax() is NaN, but not the other, the result is the non-NaN parameter. 
> IEEE-754R 표준에 따라 fminf(), fmin(), fmaxf() 또는 fmax()에 대한 입력 매개변수 중 하나가 NaN이지만 다른 변수가 아닌 경우 결과는 NaN이 아닌 매개변수입니다.

The conversion of a floating-point value to an integer value in the case where the floating-point value falls outside the range of the integer format is left undefined by IEEE-754. 
> 부동 소수점 값이 정수 형식의 범위를 벗어나는 경우 부동 소수점 값을 정수 값으로 변환하는 것은 IEEE-754에서 정의되지 않은 상태로 남겨 둡니다. 

For compute devices, the behavior is to clamp to the end of the supported range.
> 컴퓨팅 디바이스의 경우 동작은 지원되는 범위의 끝까지 고정하는 것입니다.

This is unlike the x86 architecture behavior. 
> 이는 x86 아키텍처 동작과 다릅니다.
 
http://developer.nvidia.com/content/precision-performance-floating-point-andieee-754-compliance-nvidia-gpus includes more information on the floating point accuracy and compliance of NVIDIA GPUs. 
> http://developer.nvidia.com/content/precision-performance-floating-point-andieee-754-compliance-nvidia-gpus에는 NVIDIA GPU의 부동 소수점 정확성 및 준수에 대한 추가 정보가 포함되어 있습니다.

## F.3 Compute Capability 1.x 
> F.3 컴퓨팅 기능 1.x

## F.3.1 Architecture 
> F.3.1 아키텍처

For devices of compute capability 1.x, a multiprocessor consists of:  
> 컴퓨팅 기능 1.x의 디바이스인 경우 다중프로세서는 다음으로 구성됩니다.

8 CUDA cores for arithmetic operations (see Section 5.4.1 for throughputs of arithmetic operations),  
> 산술 연산을 위한 8 개의 CUDA 코어 (산술 연산의 처리량에 대해서는 5.4.1 절 참조),

1 double-precision floating-point unit for double-precision floating-point arithmetic operations,  
> 배 정밀도 부동 소수점 연산을 위한 배 정밀도 부동 소수점 단위 1 개,

2 special function units for single-precision floating-point transcendental functions (these units can also handle single-precision floating-point multiplications),  1 warp scheduler. 
> 단 정밀도 부동 소수점 초월 함수용 (이 단위는 단 정밀도 부동 소수점 곱셈도 처리할 수 있음)은 2 개의 특수 함수 단위, 1 개의 워프 스케줄러입니다.

To execute an instruction for all threads of a warp, the warp scheduler must therefore issue the instruction over:  
> 워프의 모든 스레드에 대한 명령을 실행하려면 워프 스케줄러가 다음 명령을 발행해야 합니다.

4 clock cycles for an integer or single-precision floating-point arithmetic instruction,  32 clock cycles for a double-precision floating-point arithmetic instruction,  16 clock cycles for a single-precision floating-point transcendental instruction. 
> 정수 또는 단 정밀도 부동 소수점 산술 명령어의 경우 4 시간 사이클, 배 정밀도 부동 소수점 산술 명령어의 경우 32 시간 사이클, 단 정밀도 부동 소수점 명령어의 경우 16 시간 사이클입니다.

A multiprocessor also has a read-only constant cache that is shared by all functional units and speeds up reads from the constant memory space, which resides in device memory. Multiprocessors are grouped into Texture Processor Clusters (TPCs). 
> 또한 다중프로세서에는 모든 기능 유닛이 공유하는 읽기 전용 상수 캐시가 있으며 디바이스  메모리에 상주하는 상수 메모리 공간에서 읽기 속도가 빨라집니다. 다중프로세서는 TPC (Texture Processor Clusters)로 그룹화됩니다.

The number of multiprocessors per TPC is:  2 for devices of compute capabilities 1.0 and 1.1,  3 for devices of compute capabilities 1.2 and 1.3. 
> TPC 당 다중프로세서 수는 컴퓨팅 기능이 1.0 및 1.1인 디바이스의 경우 2, 컴퓨팅 기능 1.2 및 1.3의 디바이스인 경우 3입니다.

Each TPC has a read-only texture cache that is shared by all multiprocessors and speeds up reads from the texture memory space, which resides in device memory. 
> 각 TPC에는 모든 다중프로세서가 공유하는 읽기 전용 텍스처 캐시가 있으며 디바이스 메모리에 있는 텍스처 메모리 공간에서 읽기 속도가 빨라집니다.

Each multiprocessor accesses the texture cache via a texture unit that implements the various addressing modes and data filtering mentioned in Section 3.2.10. 
> 각 다중프로세서는 3.2.10 절에서 언급한 다양한 주소 지정 모드와 데이터 필터링을 구현하는 텍스처 유닛을 통해 텍스처 캐시에 액세스합니다.

The local and global memory spaces reside in device memory and are not cached. 
> 로컬 및 전역 메모리 공간은 디바이스 메모리에 상주하며 캐시되지 않습니다.

## F.3.2 Global Memory 
> F.3.2 전역 메모리

A global memory request for a warp is split into two memory requests, one for each half-warp, that are issued independently. 
> 워프에 대한 전역 메모리 요청은 독립적으로 발행되는 각각의 반 워프에 대해 하나씩 2 개의 메모리 요청으로 분할됩니다.

Sections F.3.2.1 and F.3.2.2 describe how the memory accesses of threads within a half-warp are coalesced into one or more memory transactions depending on the compute capability of the device. 
> F.3.2.1 절과 F.3.2.2 절은 반 - 워프 내의 스레드의 메모리 액세스가 디바이스의 컴퓨팅 기능에 따라 하나 이상의 메모리 트랜잭션으로 병합되는 방법을 설명합니다.
 
shows some examples of global memory accesses and corresponding memory transactions based on compute capability. 
> 전역 메모리 액세스 및 컴퓨팅 기능을 기반으로 하는 해당 메모리 트랜잭션의 몇 가지 예를 보여줍니다.

The resulting memory transactions are serviced at the throughput of device memory. 
> 결과 메모리 트랜잭션은 디바이스 메모리의 처리량으로 처리됩니다. 

## F.3.2.1 Devices of Compute Capability 1.0 and 1.1
> F.3.2.1 컴퓨팅 기능 1.0 및 1.1의 디바이스

To coalesce, the memory request for a half-warp must satisfy the following conditions:  
> 병합하려면 반 워프에 대한 메모리 요청은 다음 조건을 충족해야 합니다.

The size of the words accessed by the threads must be 4, 8, or 16 bytes;  
> 스레드가 액세스하는 단어의 크기는 4, 8 또는 16 바이트여야 합니다.

If this size is:  4, all 16 words must lie in the same 64-byte segment,  8, all 16 words must lie in the same 128-byte segment,  16, the first 8 words must lie in the same 128-byte segment and the last 8 words in the following 128-byte segment;  
> 이 크기가 4인 경우 모든 16 단어는 동일한 64 바이트 세그먼트에 있어야하며, 8인 경우, 모든 16 단어는 동일한 128 바이트 세그먼트에 있어야 합니다. 16인 경우, 처음 8 단어는 동일한 128 바이트 세그먼트에 있어야 합니다. 마지막 8인 경우는 다음 128 바이트 세그먼트에 있어야 합니다.

Threads must access the words in sequence: The kth thread in the half-warp must access the kth word. 
> 스레드는 순서대로 단어에 액세스해야 합니다. 반 워프의 k번째 스레드는 k번째 단어에 액세스해야 합니다.

If the half-warp meets these requirements, a 64-byte memory transaction, a 128-byte memory transaction, or two 128-byte memory transactions are issued if the size of the words accessed by the threads is 4, 8, or 16, respectively. 
> 반 워프가 이러한 요구 사항을 충족시키면, 스레드가 액세스하는 단어의 크기가 각각 4, 8 또는 16인 경우, 64 바이트 메모리 트랜잭션, 128 바이트 메모리 트랜잭션, 또는 두 개의 128 바이트 메모리 트랜잭션이 각각 발행됩니다.

Coalescing is achieved even if the warp is divergent, i.e. there are some inactive threads that do not actually access memory. 
> 병합은 워프가 다른 경우에도 수행됩니다. 즉 실제로 메모리에 액세스하지 않는 비활성 스레드가 있습니다.

If the half-warp does not meet these requirements, 16 separate 32-byte memory transactions are issued. 
> 반 워프가 이러한 요구 사항을 충족시키지 못하면 16 개의 개별 32 바이트 메모리 트랜잭션이 발행됩니다.

## F.3.2.2 Devices of Compute Capability 1.2 and 1.3
> F.3.2.2 컴퓨팅 기능 1.2 및 1.3의 디바이스

Threads can access any words in any order, including the same words, and a single memory transaction for each segment addressed by the half-warp is issued. 
> 스레드는 동일한 단어를 포함하여 임의의 순서로 임의의 단어에 액세스할 수 있으며 반 워프로 처리되는 각 세그먼트에 대한 단일 메모리 트랜잭션이 발행됩니다.

This is in contrast with devices of compute capabilities 1.0 and 1.1 where threads need to access words in sequence and coalescing only happens if the half-warp addresses a single segment. 
> 이것은 쓰레드가 워드에 순차적으로 액세스해야 하는 컴퓨팅 기능 1.0 및 1.1의 디바이스와 반비례하며 병합하는 반 워프가 단일 세그먼트를 주소 지정하는 경우에만 발생합니다.

More precisely, the following protocol is used to determine the memory transactions necessary to service all threads in a half-warp:  
> 좀더 정확히 말하면, 다음 프로토콜은 반 워프에서 모든 쓰레드를 서비스하는 데 필요한 메모리 트랜잭션을 결정하는 데 사용됩니다.

Find the memory segment that contains the address requested by the active thread with the lowest thread ID. 
> 스레드 ID가 가장 낮은 활성 스레드가 요청한 주소가 들어있는 메모리 세그먼트를 찾습니다.

The segment size depends on the size of the words accessed by the threads:  32 bytes for 1-byte words,  64 bytes for 2-byte words,  128 bytes for 4-, 8- and 16-byte words.  
> 세그먼트 크기는 스레드가 액세스하는 단어의 크기 (1 바이트 단어의 경우 32 바이트, 2 바이트 단어의 경우 64 바이트, 4 바이트, 8 바이트 및 16 바이트 단어의 경우 128 바이트)에 따라 다릅니다.

Find all other active threads whose requested address lies in the same segment.  
> 요청된 주소가 같은 세그먼트에 있는 다른 모든 활성 스레드를 찾습니다.

Reduce the transaction size, if possible:  
> 가능한 경우 트랜잭션 크기를 줄입니다.

If the transaction size is 128 bytes and only the lower or upper half is used, reduce the transaction size to 64 bytes;  
> 트랜잭션 크기가 128 바이트이고 하위 또는 상위 절반만 사용되는 경우 트랜잭션 크기를 64 바이트로 줄입니다.

If the transaction size is 64 bytes (originally or after reduction from 128 bytes) and only the lower or upper half is used, reduce the transaction size to 32 bytes. 
> 트랜잭션 크기가 64 바이트 (원래 또는 128 바이트에서 축소된 후)이고 하위 또는 상위 절반 만 사용되는 경우 트랜잭션 크기를 32 바이트로 줄입니다.
 
Carry out the transaction and mark the serviced threads as inactive.  
> 트랜잭션을 수행하고 서비스 스레드를 비활성으로 표시하십시오.

Repeat until all threads in the half-warp are serviced. 
> 반 워프의 모든 스레드가 처리될 때까지 반복하십시오.

## F.3.3 Shared Memory 
> F.3.3 공유 메모리

Shared memory has 16 banks that are organized such that successive 32-bit words map to successive banks. 
> 공유 메모리는 연속적인 32 비트 단어가 연속적인 뱅크에 매핑되도록 구성된 16 개의 뱅크를 가지고 있습니다.

Each bank has a bandwidth of 32 bits per two clock cycles. 
> 각 뱅크는 2 시간 사이클 당 32 비트의 대역폭을 가지고 있습니다.

A shared memory request for a warp is split into two memory requests, one for each half-warp, that are issued independently. 
> 워프에 대한 공유 메모리 요청은 독립적으로 발행되는 각각의 반 워프에 대해 하나씩 2 개의 메모리 요청으로 분할됩니다.

As a consequence, there can be no bank conflict between a thread belonging to the first half of a warp and a thread belonging to the second half of the same warp. 
> 결과적으로, 워프의 전반부에 속하는 스레드와 동일한 워프의 후반부에 속하는 스레드 사이에는 뱅크 충돌이 존재할 수 없습니다.

If a non-atomic instruction executed by a warp writes to the same location in shared memory for more than one of the threads of the warp, only one thread per halfwarp performs a write and which thread performs the final write is undefined. 
> 워프에 의해 실행되는 비원자 명령어가 워프의 스레드 중 하나 이상에 대해 공유 메모리의 동일한 위치에 쓰는 경우, 반 워프 당 하나의 스레드만 쓰기를 수행하고 어떤 스레드가 최종 쓰기를 수행하는지는 정의되지 않습니다.

## F.3.3.1 32-Bit Strided Access 
> F.3.3.1 32 비트 스트라이드 액세스

A common access pattern is for each thread to access a 32-bit word from an array indexed by the thread ID tid and with some strides:
> 일반적인 액세스 패턴은 각 스레드가 스레드 ID tid와 몇 가지 스트라이드로 인덱싱된 배열에서 32 비트 단어에 액세스하는 것입니다. 

In this case, threads tid and tid+n access the same bank whenever s*n is a multiple of the number of banks (i.e. 16) or, equivalently, whenever n is a multiple of 16/d where d is the greatest common divisor of 16 and s. 
> 이 경우, 스레드 tid와 tid+n은 s*n이 뱅크 수의 배수가 될 때마다 (즉 16), 또는 동일하게, n은 d가 16과 s의 가장 큰 공약수인 16/d의 배수가 될 때마다 동일한 뱅크에 접근합니다.

As a consequence, there will be no bank conflict only if half the warp size (i.e. 16) is less than or equal to 16/d., that is only if d is equal to 1, i.e. s is odd. 
> 결과적으로, 워프 크기의 절반 (즉, 16)이 16/d보다 작거나 같은 경우에만, 즉 d가 1일 때, 즉 s가 홀수인 경우에만 뱅크 충돌이 발생하지 않을 겁니다.

Figure F-2 shows some examples of strided access for devices of compute capability 3.0. 
> 그림 F-2는 컴퓨팅 기능 3.0인 디바이스에 대한 스트라이드 액세스의 몇 가지 예를 보여줍니다.

The same examples apply for devices of compute capability 1.x, but with 16 banks instead of 32. 
> 동일한 예가 컴퓨팅 기능 1.x인 디바이스에 적용되지만 32 대신 16 개의 뱅크가 적용됩니다.

Also, the access pattern for the example in the middle generates 2-way bank conflicts for devices of compute capability 1.x. 
> 또한 중간에 있는 예제의 액세스 패턴은 컴퓨팅 기능 1.x의 디바이스에 대해 양방향 뱅크 충돌을 생성합니다.

## F.3.3.2 32-Bit Broadcast Access 
> F.3.3.2 32 비트 브로드캐스트 액세스

Shared memory features a broadcast mechanism whereby a 32-bit word can be read and broadcast to several threads simultaneously when servicing one memory read request. 
> 공유 메모리는 하나의 메모리 읽기 요청을 처리할 때 동시에 32 비트 단어를 읽고 여러 스레드로 브로드캐스팅할 수 있는 브로드캐스트 메커니즘을 제공합니다.

This reduces the number of bank conflicts when several threads read from an address within the same 32-bit word. 
> 이렇게하면 여러 스레드가 같은 32 비트 단어 내의 주소에서 읽을 때 뱅크 충돌 수가 줄어듭니다.

More precisely, a memory read request made of several addresses is serviced in several steps over time by servicing one conflict-free subset of these addresses per step until all addresses have been serviced; at each step, the subset is built from the remaining addresses that have yet to be serviced using the following procedure:  
> 더 정확하게는, 몇몇 어드레스로 이루어진 메모리 읽기 요청은 모든 주소가 처리될 때까지 단계마다 이들 어드레스의 하나의 무 충돌 서브세트를 처리함으로써 시간에 따라 여러 단계로 처리됩니다; 각 단계에서, 서브세트는 다음 프로시저를 사용하여 아직 처리되지 않은 나머지 주소에서 빌드됩니다.

Select one of the words pointed to by the remaining addresses as the broadcast word;  Include in the subset:  All addresses that are within the broadcast word,  
> 나머지 주소가 가리키는 단어 중 하나를 브로드캐스트 단어로 선택합니다. 서브세트에 포함합니다. 브로드캐스트 단어 내에 있는 모든 주소,

One address for each bank (other than the broadcasting bank) pointed to by the remaining addresses. 
> 나머지 주소가 가리키는 각 뱅크 (브로드캐스트 뱅크는 제외)에 대한 하나의 주소.
 
Which word is selected as the broadcast word and which address is picked up for each bank at each cycle are unspecified. 
> 브로드캐스트 단어로 어느 단어가 선택되는지와 각 사이클의 각 뱅크에 대해 어느 주소가 선택되는지는 지정되지 않습니다. 

A common conflict-free case is when all threads of a half-warp read from an address within the same 32-bit word. 
> 공통적인 충돌없는 경우는 반 워프의 모든 스레드가 동일한 32 비트 단어 내의 주소에서 읽는 경우입니다.

Figure F-3 shows some examples of memory read accesses that involve the broadcast mechanism for devices of compute capability 3.0. 
> 그림 F-3은 컴퓨팅 기능 3.0의 디바이스에 대한 브로드캐스트 메커니즘과 관련된 메모리 읽기 액세스의 몇 가지 예를 보여줍니다.

The same examples apply for devices of compute capability 1.x, but with 16 banks instead of 32. 
> 동일한 예가 컴퓨팅 기능 1.x의 디바이스에 적용되지만 32 대신 16 개의 뱅크가 적용됩니다.

Also, the access pattern for the example at the right generates 2-way bank conflicts for devices of compute capability 1.x. 
> 또한 오른쪽 예제의 액세스 패턴은 컴퓨팅 기능 1.x의 디바이스에 대해 양방향 뱅크 충돌을 생성합니다.

## F.3.3.3 8-Bit and 16-Bit Access
> F.3.3.3 8 비트 및 16 비트 액세스

8-bit and 16-bit accesses typically generate bank conflicts. 
> 8 비트 및 16 비트 액세스는 일반적으로 뱅크 충돌을 생성합니다.

For example, there are bank conflicts if an array of char is accessed the following way: 
> 예를 들어 char 배열에 다음과 같은 방식으로 액세스하면 뱅크 충돌이 발생합니다.

because shared[0], shared[1], shared[2], and shared[3], for example, belong to the same bank. 
> 예를 들어, 공유[0], 공유[1], 공유[2] 및 공유[3]은 동일한 뱅크에 속하기 때문에.

There are no bank conflicts however, if the same array is accessed the following way: 
> 같은 배열에 다음과 같은 방법으로 액세스하면 뱅크 충돌은 없습니다. 

## F.3.3.4 Larger Than 32-Bit Access 
> F.3.3.4 32 비트 액세스보다 더 큰

Accesses that are larger than 32-bit per thread are split into 32-bit accesses that typically generate bank conflicts. 
> 스레드 당 32 비트보다 큰 액세스는 일반적으로 뱅크 충돌을 생성하는 32 비트 액세스로 분할됩니다.

For example, there are 2-way bank conflicts for arrays of doubles accessed as follows: 
> 예를 들어, 다음과 같이 액세스되는 복식 배열에 대한 양방향 뱅크 충돌이 있습니다.

as the memory request is compiled into two separate 32-bit requests with a stride of two. 
> 메모리 요구가 2 개의 스트라이드를 갖는 2 개의 별개 32 비트 요구로 컴파일되기 때문입니다.

One way to avoid bank conflicts in this case is two split the double operands like in the following sample code: 
> 이 경우 뱅크 충돌을 피하는 한 가지 방법은 다음 샘플 코드에서와 같이 두 개의 피연산자를 두 번 분할하는 것입니다.
 
This might not always improve performance however and does perform worse on devices of compute capabilities 2.x and higher. The same applies to structure assignments. 
> 하지만 성능이 항상 향상되는 것은 아니며 컴퓨팅 기능이 2.x 이상인 디바이스에서 성능이 저하될 수 있습니다. 구조 지정에도 동일하게 적용됩니다.

The following code, for example:  
> 예를 들면, 다음 코드와 같습니다.

Three separate reads without bank conflicts if type is defined as struct type {  float x, y, z; }; since each member is accessed with an odd stride of three 32-bit words;  
> 유형이 struct 유형으로 정의된 경우 {float x, y, z; }; 각 멤버는 세 개의 32 비트 단어의 홀수 스트라이드로 액세스되기 때문에; 뱅크 충돌이 없는 세 개의 개별 읽기

Two separate reads with bank conflicts if type is defined as struct type {  float x, y; }; 
since each member is accessed with an even stride of two 32-bit words. 
> 유형이 struct 유형으로 정의된 경우 {float x, y; }; 각 멤버는 두 개의 32 비트 단어의 짝수 스트라이드로 액세스되기 때문에; 뱅크 충돌이 있는 두 개의 개별 읽기

## F.4 Compute Capability 2.x 
> F.4 컴퓨팅 기능 2.x

## F.4.1 Architecture 
> F.4.1 아키텍처

For devices of compute capability 2.x, a multiprocessor consists of:  
> 컴퓨팅 기능 2.x의 디바이스인 경우 다중프로세서는 다음으로 구성됩니다.

For devices of compute capability 2.0:  32 CUDA cores for arithmetic operations (see Section 5.4.1 for throughputs of arithmetic operations),  4 special function units for single-precision floating-point transcendental functions,  
> 컴퓨팅 기능 2.0의 디바이스인 경우, 산술 연산을 위한 32 개의 CUDA 코어 (산술 연산의 처리량에 대해서는 5.4.1 절 참조), 단 정밀도 부동 소수점 초월 함수를 위한 4 개의 특수 함수 단위,

For devices of compute capability 2.1:  48 CUDA cores for arithmetic operations (see Section 5.4.1 for throughputs of arithmetic operations),  8 special function units for single-precision floating-point transcendental functions,  2 warp schedulers.  
> 컴퓨팅 기능 2.1의 디바이스인 경우, 산술 연산을 위한 48 개의 CUDA 코어 (산술 연산의 처리량에 대해서는 5.4.1 절 참조), 단 정밀도 부동 소수점 초월 함수를 위한 8 개의 특수 함수 단위, 2 개의 워프 스케줄러. 

At every instruction issue time, each scheduler issues:
> 모든 명령 발행 시간에 각 스케줄러는 다음을 발행합니다.

One instruction for devices of compute capability 2.0,  Two independent instructions for devices of compute capability 2.1, for some warp that is ready to execute, if any. 
> 컴퓨팅 기능 2.0의 디바이스인 경우에는 하나의 명령, 실행 준비가 된 일부 워프에 대한 컴퓨팅 기능 2.1의 디바이스인 경우에는 두 개의 독립적인 명령이 있습니다. 

The first scheduler is in charge of the warps with an odd ID and the second scheduler is in charge of the warps with an even ID. 
> 첫 번째 스케줄러는 홀수 ID를 갖는 워프를 담당하고, 두 번째 스케줄러는 짝수 ID를 갖는 워프를 담당합니다.

Note that when a scheduler issues a double-precision floating-point instruction, the other scheduler cannot issue any instruction. 
> 스케줄러가 배 정밀도 부동 소수점 명령어를 발행하면 다른 스케줄러는 어떤 명령어도 발행할 수 없다는 점에 유의하십시오.

A warp scheduler can issue an instruction to only half of the CUDA cores. 
> 워프 스케줄러는 CUDA 코어의 절반에만 명령을 내릴 수 있습니다.

To execute an instruction for all threads of a warp, a warp scheduler must therefore issue the instruction over two clock cycles for an integer or floating-point arithmetic instruction. 
> 워프의 모든 스레드에 대한 명령을 실행하기 위해, 워프 스케줄러는 정수 또는 부동 소수점 산술 명령에 대해 2 시간 사이클 동안 명령을 발행해야 합니다.

A multiprocessor also has a read-only constant cache that is shared by all functional units and speeds up reads from the constant memory space, which resides in device memory. 
> 또한 다중프로세서에는 모든 기능 유닛이 공유하는 읽기 전용 상수 캐시가 있으며 디바이스 메모리에 상주하는 상수 메모리 공간에서 읽기 속도가 빨라집니다.

There is an L1 cache for each multiprocessor and an L2 cache shared by all multiprocessors, both of which are used to cache accesses to local or global memory, including temporary register spills. 
> 각 다중프로세서에 대한 L1 캐시와 모든 다중프로세서에 의해 공유된 L2 캐시가 있으며, 둘 다 임시 레지스터 유출을 포함하여 로컬 또는 전역 메모리에 대한 액세스를 캐시하는 데 사용됩니다.

The cache behavior (e.g. whether reads are cached in both L1 and L2 or in L2 only) can be partially configured on a peraccess basis using modifiers to the load or store instruction. 
> 캐시 동작 (예를 들어 읽기가 L1 및 L2 또는 L2에서만 캐시되는지 여부)은 로드 또는 저장 명령어에 대한 수정자를 사용하여 액세스 기반 단위로 부분적으로 구성될 수 있습니다.

The same on-chip memory is used for both L1 and shared memory: 
> 동일한 온칩 메모리가 L1 및 공유 메모리에 모두 사용됩니다.

It can be configured as 48 KB of shared memory and 16 KB of L1 cache or as 16 KB of shared memory and 48 KB of L1 cache, using cudaFuncSetCacheConfig()/cuFuncSetCacheConfig(): 
> cudaFuncSetCacheConfig()/cuFuncSetCacheConfig()를 사용하여 공유 메모리의 48KB 및 L1 캐시의 16 KB 또는 공유 메모리의 16KB, L1 캐시의 48KB로 구성될 수 있습니다.

The default cache configuration is "prefer none," meaning "no preference." 
> 기본 캐시 구성은 "prefer none(선호하지 않음)"이며 "선호도 없음"을 의미합니다.

If a kernel is configured to have no preference, then it will default to the preference of the current thread/context, which is set using cudaDeviceSetCacheConfig()/cuCtxSetCacheConfig() (see the reference manual for details). 
> 커널이 선호도(우선권)를 가지지 않는다면, cudaDeviceSetCacheConfig()/cuCtxSetCacheConfig()를 사용하여 설정한 현재 쓰레드/컨텍스트의 선호도(우선권)를 기본값으로 하게됩니다 (자세한 내용은 참조 매뉴얼 참조).

If the current thread/context also has no preference (which is again the default setting), then whichever cache configuration was most recently used for any kernel will be the one that is used, unless a different cache configuration is required to launch the kernel (e.g., due to shared memory requirements). 
> 현재 스레드/컨텍스트에도 우선권이 없는 경우 (다시 기본 설정임), 커널을 시작하기 위해 다른 캐시 구성이 필요하지 않으면 가장 최근에 사용된 캐시 구성이 사용될 것입니다 (예를 들어, 공유 메모리의 요구 사항으로 인해)

The initial configuration is 48 KB of shared memory and 16 KB of L1 cache. 
> 초기 구성은 공유 메모리의 48KB와 L1 캐시의 16KB입니다.

Applications may query the L2 cache size by checking the l2CacheSize device property (see Section 3.2.6.1). The maximum L2 cache size is 768 KB.
> 애플리케이션은 l2CacheSize 디바이스 속성 (3.2.6.1 절 참조)을 확인하여 L2 캐시 크기를 쿼리할 수 있습니다. 최대 L2 캐시 크기는 768KB입니다.

Multiprocessors are grouped into Graphics Processor Clusters (GPCs). A GPC includes four multiprocessors. 
> 다중프로세서는 GPC (Graphics Processor Clusters)로 그룹화됩니다. GPC는 4 개의 다중프로세서를 포함합니다.

Each multiprocessor has a read-only texture cache to speed up reads from the texture memory space, which resides in device memory. 
> 각 다중프로세서에는 읽기 전용 텍스처 캐시가 있어 디바이스 메모리에 있는 텍스처 메모리 공간에서 읽기 속도를 높입니다.

It accesses the texture cache via a texture unit that implements the various addressing modes and data filtering mentioned in Section 3.2.10. 
> 3.2.10 절에서 언급한 다양한 어드레싱 모드와 데이터 필터링을 구현하는 텍스처 유닛을 통해 텍스처 캐시에 접근합니다.

## F.4.2 Global Memory 
> F.4.2 전역 메모리

Global memory accesses are cached. 
> 전역 메모리 액세스는 캐시됩니다.

Using the –dlcm compilation flag, they can be configured at compile time to be cached in both L1 and L2 (-Xptxas -dlcm=ca) (this is the default setting) or in L2 only (-Xptxas -dlcm=cg). 
> -dlcm 컴파일 플래그를 사용하면 컴파일 시 L1 및 L2 (-Xptxas -dlcm=ca) (이는 기본 설정입니다) 또는 L2 전용 (-Xptxas -dlcm = cg)에 캐시되도록 구성할 수 있습니다.

A cache line is 128 bytes and maps to a 128-byte aligned segment in device memory. 
> 캐시 라인은 128 바이트이며 디바이스 메모리의 128 바이트 정렬 세그먼트에 매핑됩니다.

Memory accesses that are cached in both L1 and L2 are serviced with 128-byte memory transactions whereas memory accesses that are cached in L2 only are serviced with 32-byte memory transactions. 
> L1과 L2 모두에 캐싱되는 메모리 액세스는 128 바이트 메모리 트랜잭션으로 처리되지만 L2에만 캐시된 메모리 액세스는 32 바이트 메모리 트랜잭션으로 처리됩니다.

Caching in L2 only can therefore reduce over-fetch, for example, in the case of scattered memory accesses. 
> 따라서 L2로만 캐시하면 흩어져있는 메모리 액세스의 경우에는 오버페치(과도한 가져오기)를 줄일 수 있습니다.
 
If the size of the words accessed by each thread is more than 4 bytes, a memory request by a warp is first split into separate 128-byte memory requests that are issued independently:  
> 각 스레드에 의해 액세스된 워드의 크기가 4 바이트보다 크다면, 워프에 의한 메모리 요청은 먼저 독립적으로 발행되는 별도의 128 바이트 메모리 요청으로 분할됩니다.

Two memory requests, one for each half-warp, if the size is 8 bytes,  
> 크기가 8 바이트인 경우, 두 개의 메모리 요청, 각 반 워프마다 하나입니다.

Four memory requests, one for each quarter-warp, if the size is 16 bytes. 
> 크기가 16 바이트인 경우, 4 개의 메모리 요청, 각 쿼트(4분의 1) 워프마다 하나입니다.

Each memory request is then broken down into cache line requests that are issued independently. 
> 각 메모리 요청은 독립적으로 발행되는 캐시 라인 요청으로 분류됩니다.

A cache line request is serviced at the throughput of L1 or L2 cache in case of a cache hit, or at the throughput of device memory, otherwise. 
> 캐시 라인 요청은 캐시 히트의 경우 L1 또는 L2 캐시의 처리량에서, 그렇지 않으면 디바이스 메모리의 처리량에서 처리됩니다.

Note that threads can access any words in any order, including the same words. 
> 스레드는 동일한 단어를 포함하여 어떤 순서로든 어떤 단어에도 액세스할 수 있습니다.

If a non-atomic instruction executed by a warp writes to the same location in global memory for more than one of the threads of the warp, only one thread performs a write and which thread does it is undefined.
> 워프에 의해 실행되는 비원자 명령어가 전역 메모리의 동일한 위치에 워프의 스레드 중 하나 이상에 대해 쓰면 하나의 스레드만 쓰기를 수행하고 어떤 스레드는 정의되지 않습니다.

Figure F-1 shows some examples of global memory accesses and corresponding memory transactions based on compute capability. 
> 그림 F-1은 전역 메모리 액세스 및 컴퓨팅 기능을 기반으로 하는 해당 메모리 트랜잭션의 몇 가지 예를 보여줍니다.

## F.4.3 Shared Memory 
> F.4.3 공유 메모리

Shared memory has 32 banks that are organized such that successive 32-bit words map to successive banks. 
> 공유 메모리는 연속적인 32 비트 워드가 연속적인 뱅크에 매핑되도록 구성된 32 개의 뱅크를 가지고 있습니다.

Each bank has a bandwidth of 32 bits per two clock cycles. 
> 각 뱅크는 2 시간 사이클 당 32 비트의 대역폭을 가지고 있습니다.

A shared memory request for a warp does not generate a bank conflict between two threads that access any address within the same 32-bit word (even though the two addresses fall in the same bank): 
> 워프에 대한 공유 메모리 요청은 같은 32 비트 단어 내의 모든 주소에 액세스하는 두 스레드 간에 뱅크 충돌을 생성하지 않습니다 (두 주소가 동일한 뱅크에 있음에도 불구하고).

In that case, for read accesses, the word is broadcast to the requesting threads (and unlike for devices of compute capability 1.x, multiple words can be broadcast in a single transaction) and for write accesses, each address is written by only one of the threads (which thread performs the write is undefined). 
> 이 경우에는 읽기 액세스의 경우 워드는 요청 스레드로 브로드캐스팅되며 (컴퓨팅 기능 1.x의 디바이스와 달리 단일 트랜잭션에서 여러 단어가 브로드캐스팅될 수 있음) 쓰기 액세스의 경우 각 주소는 스레드 중 하나에서만 쓰여집니다 (쓰기를 수행하는 스레드는 정의되지 않음).

This means, in particular, that unlike for devices of compute capability 1.x, there are no bank conflicts if an array of char is accessed as follows, for example: 
> 즉, 컴퓨팅 기능 1.x의 디바이스와 달리 char 배열에 다음과 같이 액세스하면 뱅크 충돌이 발생하지 않습니다.

Also, unlike for devices of compute capability 1.x, there may be bank conflicts between a thread belonging to the first half of a warp and a thread belonging to the second half of the same warp. 
> 또한, 컴퓨팅 기능 1.x의 디바이스와 달리, 워프의 전반부에 속하는 스레드와 동일한 워프의 후반부에 속하는 스레드간에 뱅크 충돌이 있을 수 있습니다.

Figure F-3 shows some examples of memory read accesses that involve the broadcast mechanism for devices of compute capability 3.0. 
> 그림 F-3은 컴퓨팅 기능 3.0의 디바이스에 대한 브로드캐스트 메커니즘과 관련된 메모리 읽기 액세스의 몇 가지 예를 보여줍니다.

The same examples apply for devices of compute capability 2.x. 
> 컴퓨팅 기능 2.x의 디바이스에도 동일한 예제가 적용됩니다.

## F.4.3.1 32-Bit Strided Access 
> F.4.3.1 32 비트 스트라이드 액세스

A common access pattern is for each thread to access a 32-bit word from an array indexed by the thread ID tid and with some stride s: 
> 일반적인 액세스 패턴은 각 스레드가 스레드 ID tid 및 일부 스트라이드로 인덱싱된 배열에서 32 비트 워드에 액세스하는 것입니다. 

In this case, threads tid and tid+n access the same bank whenever s*n is a multiple of the number of banks (i.e. 32) or, equivalently, whenever n is a multiple of 32/d where d is the greatest common divisor of 32 and s. 
> 이 경우에는, tid와 tid+n은 s*n이 뱅크 수의 배수가 될 때마다 (즉, 32), 또는 n이 d가 32와 s의 가장 큰 공약수가 되는 32/d의 배수가 될 때마다 동일한 뱅크에 액세스합니다.

As a consequence, there will be no bank conflict only if the warp size (i.e. 32) is less than or equal to 32/d., that is only if d is equal to 1, i.e. s is odd. 
> 결과적으로, 워프 크기 (즉, 32)가 32/d 이하인 경우에만, 즉 d가 1과 같을 때, 즉 s가 홀수인 경우에만 뱅크 충돌이 발생하지 않습니다.

Figure F-2 shows some examples of strided access for devices of compute capability 3.0. The same examples apply for devices of compute capability 2.x. 
> 그림 F-2는 컴퓨팅 기능 3.0에 대한 스트라이드 액세스의 몇 가지 예를 보여줍니다. 컴퓨팅 기능 2.x의 디바이스에도 동일한 예가 적용됩니다.

However, the access pattern for the example in the middle generates 2-way bank conflicts for devices of compute capability 2.x. 
> 그러나 중간에 있는 예제의 액세스 패턴은 컴퓨팅 기능 2.x의 디바이스에 대해 양방향 뱅크 충돌을 생성합니다.

## F.4.3.2 Larger Than 32-Bit Access
> F.4.3.2 32 비트 액세스보다 큰

64-bit and 128-bit accesses are specifically handled to minimize bank conflicts as described below. 
> 64 비트 및 128 비트 액세스는 아래에 설명된 것처럼 뱅크 충돌을 최소화하기 위해 특별히 처리됩니다.

Other accesses larger than 32-bit are split into 32-bit, 64-bit, or 128-bit accesses. 
> 32 비트보다 큰 다른 액세스는 32 비트, 64 비트 또는 128 비트 액세스로 분할됩니다.

The following code, for example: 
> 예를 들어 다음 코드와 같습니다: 

results in three separate 32-bit reads without bank conflicts since each member is accessed with a stride of three 32-bit words. 
> 3 개의 32 비트 단어의 스트라이드를 사용하여 각 멤버에 액세스하므로 뱅크 충돌없이 3 개의 개별 32 비트 읽기가 수행됩니다.

64-Bit Accesses 
> 64 비트 액세스

For 64-bit accesses, a bank conflict only occurs if two threads in either of the halfwarps access different addresses belonging to the same bank. 
> 64 비트 액세스의 경우 반 워프 중 하나에 두 개의 스레드가 동일한 뱅크에 속한 다른 주소에 액세스하는 경우에만 은행 충돌이 발생합니다.

Unlike for devices of compute capability 1.x, there are no bank conflicts for arrays of doubles accessed as follows, for example: 
> 컴퓨팅 기능 1.x의 디바이스와 달리 다음과 같이 액세스되는 복식 배열에 대한 뱅크 충돌은 없습니다.

128-Bit Accesses 
> 128 비트 액세스

The majority of 128-bit accesses will cause 2-way bank conflicts, even if no two threads in a quarter-warp access different addresses belonging to the same bank.  
> 쿼터 - 워프의 두 스레드가 동일한 뱅크에 속한 다른 주소에 액세스하지 않더라도 대부분의 128 비트 액세스는 양방향 뱅크 충돌을 야기합니다.

Therefore, to determine the ways of bank conflicts, one must add 1 to the maximum number of threads in a quarter-warp that access different addresses belonging to the same bank. 
> 따라서 뱅크 충돌의 방법을 결정하려면 동일한 뱅크에 속한 다른 주소에 액세스하는 쿼터-워프의 최대 스레드 수에 1을 더해야합니다.

## F.4.4 Constant Memory 
> F.4.4 상수 메모리

In addition to the constant memory space supported by devices of all compute capabilities (where __constant__ variables reside), devices of compute capability 2.x support the LDU (LoaD Uniform) instruction that the compiler uses to load any variable that is:  
> 컴퓨팅 기능 (__constant__ 변수가 있는)의 디바이스에서 지원되는 상수 메모리 공간 외에 컴퓨팅 기능 2.x 인 디바이스는 컴파일러가 다음과 같은 변수를 로드하는 데 사용하는 LDU (LoaD Uniform) 명령을 지원합니다.

pointing to global memory,  read-only in the kernel (programmer can enforce this using the const keyword),  not dependent on thread ID. 
> 전역 메모리를 가리키며, 스레드 ID에 의존하지 않고 커널에서 읽기 전용 (프로그래머는 const 키워드를 사용하여 이것을 시행할 수 있음).
 
## F.5 Compute Capability 3.0 
> F.5 컴퓨팅 기능 3.0

## F.5.1 Architecture 
> F.5.1 아키텍처

A multiprocessor consists of:  192 CUDA cores for arithmetic operations (see Section 5.4.1 for throughputs of arithmetic operations),  32 special function units for single-precision floating-point transcendental functions,  4 warp schedulers. 
> 다중프로세서는 산술 연산을 위한 192 개의 CUDA 코어 (산술 연산의 처리량에 대해서는 5.4.1 절 참조), 단 정밀도 부동 소수점 초월 함수를 위한 32 개의 특수 함수 유닛, 4 개의 워프 스케줄러로 구성됩니다.

When a multiprocessor is given warps to execute, it first distributes them among the four schedulers. 
> 워프가 실행하도록 다중프로세서가 주어지면 먼저 4 개의 스케줄러에 다중프로세서를 배포합니다.

Then, at every instruction issue time, each scheduler issues two independent instructions for one of its assigned warps that is ready to execute, if any. 
> 그런 다음, 모든 명령어 발행 시간에 각 스케줄러는 할당된 워프 중 하나에 대해 실행 준비가 된 두 개의 독립적인 명령어를 발행합니다.

A multiprocessor has a read-only constant cache that is shared by all functional units and speeds up reads from the constant memory space, which resides in device memory. 
> 다중프로세서에는 모든 기능 유닛이 공유하는 읽기 전용 상수 캐시가 있으며 디바이스 메모리에 있는 상수 메모리 공간에서 읽기 속도가 빨라집니다.

There is an L1 cache for each multiprocessor and an L2 cache shared by all multiprocessors, both of which are used to cache accesses to local or global memory, including temporary register spills. 
> 각 다중프로세서에 대한 L1 캐시와 모든 다중프로세서에 의해 공유된 L2 캐시가 있으며, 둘 다 임시 레지스터 유출을 포함하여 로컬 또는 전역 메모리에 대한 액세스를 캐시하는 데 사용됩니다.

The cache behavior (e.g. whether reads are cached in both L1 and L2 or in L2 only) can be partially configured on a peraccess basis using modifiers to the load or store instruction. 
> 캐시 동작 (예를 들어 읽기가 L1 및 L2 또는 L2에서만 캐시되는지 여부)은 로드 또는 저장 명령어에 대한 수정자를 사용하여 액세스 기반 단위로 부분적으로 구성될 수 있습니다.

The same on-chip memory is used for both L1 and shared memory: 
> 동일한 온칩 메모리가 L1 및 공유 메모리에 모두 사용됩니다.

It can be configured as 48 KB of shared memory and 16 KB of L1 cache or as 16 KB of shared memory and 48 KB of L1 cache or as 32 KB of shared memory and 32 KB of L1 cache, using cudaFuncSetCacheConfig()/cuFuncSetCacheConfig(): 
> cudaFuncSetCacheConfig()/cuFuncSetCacheConfig()를 사용하여 공유 메모리의 48KB와 L1 캐시의 16KB 또는 공유 메모리의 16KB와 L1 캐시의 48KB 또는 공유 메모리의 32KB와 L1 캐시의 32KB로 구성될 수 있습니다. ) 

The default cache configuration is "prefer none," meaning "no preference." 
> 기본 캐시 구성은 "선호하지 않음"은 "선호도 없음"을 의미합니다.

If a kernel is configured to have no preference, then it will default to the preference of the current thread/context, which is set using cudaDeviceSetCacheConfig()/cuCtxSetCacheConfig() (see the reference manual for details). 
> 커널이 선호도(우선권)을 가지지 않는다면, cudaDeviceSetCacheConfig()/cuCtxSetCacheConfig()를 사용하여 설정한 현재 쓰레드/컨텍스트의 우선권을 기본값으로 사용하게 됩니다 (자세한 내용은 참조 매뉴얼 확인).

If the current thread/context also has no preference (which is again the default setting), then whichever cache configuration was most recently used for any kernel will be the one that is used, unless a different cache configuration is required to launch the kernel (e.g., due to shared memory requirements). 
> 현재 스레드/컨텍스트에도 우선권을 가지지 않는 경우 (다시 기본 설정임) 커널을 시작하기 위해 다른 캐시 구성이 필요하지 않으면 어떤 캐시 구성이든 가장 최근에 사용된 커널의 사용됩니다(예를 들어 공유 메모리 요구 사항으로 인해).

The initial configuration is 48 KB of shared memory and 16 KB of L1 cache. 
> 초기 구성은 공유 메모리의 48KB와 L1 캐시의 16KB입니다.

Applications may query the L2 cache size by checking the l2CacheSize device property (see Section 3.2.6.1). 
> 애플리케이션은 l2CacheSize 디바이스 속성 (3.2.6.1 절 참조)을 확인하여 L2 캐시 크기를 쿼리할 수 있습니다.

The maximum L2 cache size is 512 KB. 
> 최대 L2 캐시 크기는 512KB입니다. 

Multiprocessors are grouped into Graphics Processor Clusters (GPCs). A GPC includes two multiprocessors. 
> 다중 프로세서는 GPC (Graphics Processor Clusters)로 그룹화됩니다. GPC는 두 개의 다중프로세서를 포함합니다.

Each multiprocessor has a read-only texture cache to speed up reads from the texture memory space, which resides in device memory. 
> 각 다중프로세서에는 읽기 전용 텍스처 캐시가 있어 디바이스 메모리에 있는 텍스처 메모리 공간에서 읽기 속도를 높입니다.

It accesses the texture cache via a texture unit that implements the various addressing modes and data filtering mentioned in Section 3.2.10. 
> 3.2.10 절에서 언급한 다양한 주소 지정 모드와 데이터 필터링을 구현하는 텍스처 유닛을 통해 텍스처 캐시에 접근합니다.

## F.5.2 Global Memory 
> F.5.2 전역 메모리

Global memory accesses for devices of compute capability 3.0 behave in the same way as for devices of compute capability 2.x (see Section F.4.2). 
> 컴퓨팅 기능 3.0의 디바이스에 대한 전역 메모리 액세스는 컴퓨팅 기능 2.x의 디바이스와 동일한 방식으로 작동합니다 (F.4.2 절 참조).

Figure F-1 shows some examples of global memory accesses and corresponding memory transactions based on compute capability. 
> 그림 F-1은 전역 메모리 액세스 및 계산 기능을 기반으로하는 해당 메모리 트랜잭션의 몇 가지 예를 보여줍니다.
 
Figure F-1. Examples of Global Memory Accesses by a Warp, 4-Byte Word per Thread, and Associated Memory Transactions Based on Compute Capability 
> 그림 F-1. 워프에 의한 전역 메모리 액세스의 예, 스레드 당 4 바이트 단어, 및 컴퓨팅 기능을 기반으로 하는 관련 메모리 트랜잭션

## F.5.3 Shared Memory 
> F.5.3 공유 메모리

Shared memory has 32 banks with two addressing modes that are described below. 
> 공유 메모리는 아래에 설명된 두 개의 주소 지정 모드가 있는 32 개의 뱅크를 가지고 있습니다.

The addressing mode can be queried using cudaDeviceGetSharedMemConfig() and set using cudaDeviceSetSharedMemConfig() (see reference manual for more details). 
> 주소 지정 모드는 cudaDeviceGetSharedMemConfig()를 사용하여 쿼리하고 cudaDeviceSetSharedMemConfig()를 사용하여 설정할 수 있습니다 (자세한 내용은 참조 메뉴얼 확인).

Each bank has a bandwidth of 64 bits per clock cycle.  Figure F-2 shows some examples of strided access. 
> 각 뱅크는 시간 사이클 당 64 비트의 대역폭을 가지고 있습니다. 그림 F-2는 스트라이드 액세스의 몇 가지 예를 보여줍니다.

Figure F-3 shows some examples of memory read accesses that involve the broadcast mechanism. 
> 그림 F-3은 브로드캐스트 메커니즘과 관련된 메모리 읽기 액세스의 몇 가지 예를 보여줍니다. 

## F.5.3.1 64-Bit Mode 
> F.5.3.1 64 비트 모드

Successive 64-bit words map to successive banks.
> 연속적인 64 비트 워드는 연속적인 뱅크에 매핑됩니다.

A shared memory request for a warp does not generate a bank conflict between two threads that access any address within the same 64-bit word (even though the two addresses fall in the same bank): 
> 워프에 대한 공유 메모리 요청은 같은 64 비트 워드 내의 임의의 주소에 액세스하는 두 스레드 간에 뱅크 충돌을 생성하지 않습니다 (두 주소가 동일한 뱅크에 있더라도).

In that case, for read accesses, the word is broadcast to the requesting threads and for write accesses, each address is written by only one of the threads (which thread performs the write is undefined). 
> 이 경우에는 읽기 액세스의 경우 해당 단어가 요청 스레드로 브로드캐스팅되고 쓰기 액세스의 경우 각 주소는 스레드 중 하나에서만 기록됩니다 (쓰기를 수행하는 스레드는 정의되지 않음).

In this mode, the same access pattern generates fewer bank conflicts than on devices of compute capability 2.x for 64-bit accesses and as many or fewer for 32-bit accesses. 
> 이 모드에서 동일한 액세스 패턴은 64 비트 액세스의 컴퓨팅 기능 2.x 및 32 비트 액세스의 여러 가지 기능보다 더 적은 뱅크 충돌을 생성합니다.

## F.5.3.2 32-Bit Mode 
> F.5.3.2 32 비트 모드 

Successive 32-bit words map to successive banks.
> 연속적인 32 비트 워드는 연속적인 뱅크에 매핑됩니다.

A shared memory request for a warp does not generate a bank conflict between two threads that access any address within the same 32-bit word or within two 32-bit words whose indices i and j are in the same 64-word aligned segment (i.e. a segment whose first index is a multiple of 64) and such that j=i+32 (even though the two addresses fall in the same bank): 
> 워프에 대한 공유 메모리 요청은 동일한 32 비트 워드 내에서 또는 인덱스 i와 j가 동일한 64 워드로 정렬된 세그먼트 (즉, 첫 번째 인덱스가 64의 배수인 세그먼트)와  j=i+32 (두 개의 주소가 동일한 뱅크에 있더라도) 같은 2 개의 32 비트 워드 내의 임의의 어드레스에 액세스하는 2 개의 스레드 간에 뱅크 충돌을 생성하지 않습니다..

In that case, for read accesses, the 32-bit words are broadcast to the requesting threads and for write accesses, each address is written by only one of the threads (which thread performs the write is undefined). 
> 이 경우에는 읽기 액세스의 경우 32 비트 워드가 요청 스레드로 브로드캐스팅되고 쓰기 액세스의 경우 각 주소는 스레드 중 하나에서만 기록됩니다 (쓰기를 수행하는 스레드는 정의되지 않음).

In this mode, the same access pattern generates as many or fewer bank conflicts than on devices of compute capability 2.x.   
> 이 모드에서 동일한 액세스 패턴은 컴퓨팅 기능 2.x의 디바이스와 비교하여 많거나 적은 뱅크 충돌을 생성합니다. 
 
Left: Linear addressing with a stride of one 32-bit word (no bank conflict). 
> 왼쪽: 하나의 32 비트 워드 (뱅크 충돌 없음)의 스트라이드를 사용하여 선형 주소 지정.

Middle: Linear addressing with a stride of two 32-bit words (no bank conflict). 
> 중간: 2 개의 32 비트 워드 (뱅크 충돌 없음)의 스트라이드를 사용하여 선형 주소 지정.

Right: Linear addressing with a stride of three 32-bit words (no bank conflict). 
> 오른쪽: 3 개의 32 비트 워드 (뱅크 충돌 없음)의 스트라이드를 사용하여 선형 주소 지정.

Figure F-2  Examples of Strided Shared Memory Accesses for Devices of Compute Capability 3.0  
> 그림 F-2 컴퓨팅 기능 3.0의 디바이스에 대한 스트라이드 공유 메모리 액세스의 예
 
Left: Conflict-free access via random permutation. 
> 왼쪽: 무작위 순열(치환)을 통한 충돌없는 액세스.

Middle: Conflict-free access since threads 3, 4, 6, 7, and 9 access the same word within bank 5. 
> 중간: 스레드 3, 4, 6, 7 및 9가 뱅크 5 내의 동일한 단어에 액세스하기 때문에 충돌없는 액세스.

Right: Conflict-free broadcast access (threads access the same word within a bank). 
> 오른쪽: 충돌없는 브로드캐스트 액세스 (스레드는 뱅크 내에서 동일한 단어에 액세스함).

Figure F-3  Examples of Irregular Shared Memory Accesses for Devices of Compute Capability 3.0 
> 그림 F-3 컴퓨팅 기능 3.0의 디바이스에 대한 불규칙한 공유 메모리 액세스의 예
