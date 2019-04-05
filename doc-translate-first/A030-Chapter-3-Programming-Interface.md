# Chapter 3. Programming Interface 
> 3 장. 프로그래밍 인터페이스

CUDA C provides a simple path for users familiar with the C programming language to easily write programs for execution by the device. 
> CUDA C는 C 프로그래밍 언어에 익숙한 사용자에게 간단한 경로를 제공하여 디바이스로 실행하는 프로그램을 쉽게 작성할 수 있습니다.

It consists of a minimal set of extensions to the C language and a runtime library. 
> 이것은 C 언어와 런타임 라이브러리에 대한 최소한의 확장 세트로 구성됩니다.

The core language extensions have been introduced in Chapter 2. 
> 핵심 언어 확장은 2장에서 소개되었습니다.

They allow programmers to define a kernel as a C function and use some new syntax to specify the grid and block dimension each time the function is called. 
> 이들은 프로그래머가 커널을 C 함수로 정의하고 새로운 구문을 사용하여 함수가 호출될 때마다 그리드와 블록 차원(크기)을 지정합니다.

A complete description of all extensions can be found in Appendix B. 
> 모든 확장에 대한 자세한 설명은 부록 B에 있습니다.

Any source file that contains some of these extensions must be compiled with nvcc as outlined in Section 3.1. 
> 이러한 확장 중 일부가 들어있는 소스 파일은 3.1절에서 설명한 대로 nvcc로 컴파일해야 합니다.

The runtime is introduced in Section 3.2. 
> 런타임은 3.2절에서 소개됩니다.

It provides C functions that execute on the host to allocate and deallocate device memory, transfer data between host memory and device memory, manage systems with multiple devices, etc. 
> 호스트에서 실행되는 C 함수를 제공하여 디바이스 메모리를 할당 및 할당 해제하고, 호스트 메모리와 디바이스 메모리 간에 데이터를 전송하며, 다중 디바이스가 있는 시스템을 관리합니다.

A complete description of the runtime can be found in the CUDA reference manual. 
> 런타임에 대한 자세한 설명은 CUDA 참조 설명서에서 확인할 수 있습니다.

The runtime is built on top of a lower-level C API, the CUDA driver API, which is also accessible by the application. 
> 런타임은 애플리케이션이 액세스할 수 있는 CUDA 드라이버 API인 저수준 C API 상단에 구축합니다.

The driver API provides an additional level of control by exposing lower-level concepts such as CUDA contexts –the analogue of host processes for the device– and CUDA modules –the analogue of dynamically loaded libraries for the device. 
> 드라이버 API는 CUDA 컨텍스트 -디바이스 및 CUDA 모듈을 위한 호스트 프로세스의 아날로그- 및 CUDA 모듈 - 디바이스에 대해 동적으로 로드된 라이브러리의 아날로그같은 저수준의 개념을 노출함으로써 추가 제어 수준을 제공합니다.

Most applications do not use the driver API as they do not need this additional level of control and when using the runtime, context and module management are implicit, resulting in more concise code. 
> 대부분의 애플리케이션은 이 추가 제어 레벨이 필요하지 않으므로 드라이버 API를 사용하지 않으며 런타임을 사용할 때 컨텍스트 및 모듈 관리가 암시적이므로 보다 간결한 코드가 생성됩니다.

The driver API is introduced in Appendix G and fully described in the reference manual.   
> 드라이버 API는 부록 G에서 소개되고 참조 설명서에서 자세히 설명됩니다.

## 3.1 Compilation with NVCC 
> 3.1 NVCC 컴파일

Kernels can be written using the CUDA instruction set architecture, called PTX, which is described in the PTX reference manual. 
> 커널은 PTX 참조 설명서에 설명된 PTX라는 CUDA 명령어 세트 아키텍처를 사용하여 작성할 수 있습니다.

It is however usually more effective to use a high-level programming language such as C. 
> 그러나 일반적으로 C와 같은 높은 수준의 프로그래밍 언어를 사용하는 것이 더 효과적입니다.

In both cases, kernels must be compiled into binary code by nvcc to execute on the device. 
> 두 경우 모두, 디바이스에서 실행하려면 커널을 nvcc로 바이너리 코드에 컴파일해야 합니다. 

nvcc is a compiler driver that simplifies the process of compiling C or PTX code: 
> nvcc는 C 또는 PTX 코드를 컴파일하는 과정을 단순화하는 컴파일러 드라이버입니다.

It provides simple and familiar command line options and executes them by invoking the collection of tools that implement the different compilation stages. 
> 간단하고 익숙한 명령행 옵션을 제공하고 다양한 컴파일 단계를 구현하는 도구 컬렉션을 호출하여 이를 실행합니다.

This section gives an overview of nvcc workflow and command options. 
> 이 절에서는 nvcc 워크플로와 명령 옵션에 대해 간략히 설명합니다.

A complete description can be found in the nvcc user manual. 
> 자세한 설명은 nvcc 사용자 설명서에서 찾을 수 있습니다.

## 3.1.1.1 Offline Compilation 
> 3.1.1.1 오프라인 컴파일

Source files compiled with nvcc can include a mix of host code (i.e. code that executes on the host) and device code (i.e. code that executes on the device). 
> nvcc로 컴파일된 소스 파일은 호스트 코드 (즉, 호스트에서 실행되는 코드)와 디바이스 코드 (즉, 디바이스에서 실행되는 코드)의 혼합을 포함할 수 있습니다.

nvcc’s basic workflow consists in separating device code from host code and then:  
> nvcc의 기본 워크플로는 호스트 코드에서 디바이스 코드를 분리하면 다음과 같습니다. 

compiling the device code into an assembly form (PTX code) and/or binary form (cubin object),  and modifying the host code by replacing the <<<…>>> syntax introduced in Section 2.1 (and described in more details in Section B.18) by the necessary CUDA C runtime function calls to load and launch each compiled kernel from the PTX code and/or cubin object. 
> 디바이스 코드를 어셈블리(조립) 형식 (PTX 코드) 및/또는 이진 형식 (cubin 객체)으로 컴파일하고, 2.1 절에서 소개한 <<<...>>> 구문을 대체하여 호스트 코드를 수정하는 방법에 대해 설명합니다 (자세한 내용은 B.18절에 설명됨) 필요한 PTX 코드 및/또는 cubin 객체에서 컴파일된 각 커널을 로드하고 시작하는데 필요한 CUDA C 런타임 함수 호출에 의해 호출됩니다.

The modified host code is output either as C code that is left to be compiled using another tool or as object code directly by letting nvcc invoke the host compiler during the last compilation stage. 
> 수정된 호스트 코드는 다른 도구를 사용하여 컴파일할 C 코드로 출력되거나 마지막 컴파일 단계에서 nvcc가 호스트 컴파일러를 호출하도록 하여 객체 코드로 직접 출력됩니다.

Applications can then:  Either link to the compiled host code, Or ignore the modifed host code (if any) and use the CUDA driver API (see Appendix G) to load and execute the PTX code or cubin object. 
> 그런 다음 애플리케이션은 다음을 수행할 수 있습니다. 컴파일된 호스트 코드에 링크하거나, 또는 수정된 호스트 코드 (있는 경우)를 무시하고 CUDA 드라이버 API (부록 G 참조)를 사용하여 PTX 코드 또는 cubin 객체를 로드하고 실행합니다.

## 3.1.1.2 Just-in-Time Compilation 
> 3.1.1.2 Just-in-Time(즉시) 컴파일

Any PTX code loaded by an application at runtime is compiled further to binary code by the device driver. 
> 런타임에 애플리케이션으로 로드된 PTX 코드는 디바이스 드라이버에 의해 2진 코드로 더 컴파일됩니다.

This is called just-in-time compilation. Just-in-time compilation increases application load time, but allows applications to benefit from latest compiler improvements. 
> 이를 JIT (just-in-time) 컴파일이라고 합니다. JIT (Just-In-Time) 컴파일은 애플리케이션 로드 시간을 늘리지만 애플리케이션에서 최신 컴파일러 개선의 이점을 누릴 수 있습니다.

It is also the only way for applications to run on devices that did not exist at the time the application was compiled, as detailed in Section 3.1.4. 
> 또한 3.1.4 절에서 설명한 것처럼 애플리케이션을 컴파일할 때 존재하지 않는 디바이스에서 애플리케이션을 실행하는 유일한 방법이기도 합니다.

When the device driver just-in-time compiles some PTX code for some application, it automatically caches a copy of the generated binary code in order to avoid repeating the compilation in subsequent invocations of the application. 
> 디바이스 드라이버가 일부 애플리케이션에 대해 PTX 코드를 JIT 컴파일하면 애플리케이션의  후속 호출에서 컴파일을 반복하지 않도록 자동으로 생성된 이진 코드의 복사본을 캐시합니다.

The cache – referred to as compute cache – is automatically invalidated when the device driver is upgraded, so that applications can benefit from the improvements in the new just-in-time compiler built into the device driver. 
> 컴퓨팅 캐시라고하는 캐시는 디바이스 드라이버가 업그레이드될 때 자동으로 무효화되므로 애플리케이션은 디바이스 드라이버에 내장된 새로운 JIT 컴파일러의 향상을 통해 이익을 얻을 수 있습니다.

Environment variables are available to control just-in-time compilation:  
> Just-In-Time 컴파일을 제어하기 위해 환경 변수를 사용할 수 있습니다.

Setting CUDA_CACHE_DISABLE to 1 disables caching (i.e. no binary code is added to or retrieved from the cache).  
> CUDA_CACHE_DISABLE을 1로 설정하면 캐싱이 사용 중지됩니다 (즉, 2진 코드가 캐시에 추가되거나 캐시에서 검색되지 않음).

CUDA_CACHE_MAXSIZE specifies the size of the compute cache in bytes; the default size is 32 MB and the maximum size is 4 GB; binary codes whose size exceeds the cache size are not cached; older binary codes are evicted from the cache to make room for newer binary codes if needed.  
> CUDA_CACHE_MAXSIZE는 계산 캐시의 크기를 바이트 단위로 지정합니다. 기본 크기는 32MB이고 최대 크기는 4GB입니다. 크기가 캐시 크기를 초과하는 2진 코드는 캐시되지 않습니다. 필요하다면 이전 바이너리 코드가 캐시에서 제거되어 새로운 바이너리 코드를 위한 공간을 만듭니다.

CUDA_CACHE_PATH specifies the folder where the compute cache files are stored; the default values are:  
> CUDA_CACHE_PATH는 계산 캐시 파일이 저장되는 폴더를 지정합니다. 기본값은 다음과 같습니다 : 

Setting CUDA_FORCE_PTX_JIT to 1 forces the device driver to ignore any binary code embedded in an application (see Section 3.1.4) and to just-in-time compile embedded PTX code instead; if a kernel does not have embedded PTX code, it will fail to load; this environment variable can be used to validate that PTX code is embedded in an application and that its just-in-time compilation works as expected to guarantee application forward compatibility with future architectures. 
> CUDA_FORCE_PTX_JIT를 1로 설정하면 디바이스 드라이버가 애플리케이션에 포함된 이진 코드 (3.1.4 절 참조) 및 Just-In-Time 임베디드 PTX 코드를 무시합니다. 커널에 내장된 PTX 코드가 없으면 로드에 실패합니다. 이 환경 변수를 사용하여 PTX 코드가 애플리케이션에 포함되어 있는지 확인하고 JIT (just-in-time) 컴파일이 예상대로 작동하여 애플리케이션과 향후 아키텍처 간의 호환성을 보장할 수 있습니다.

## 3.1.2 Binary Compatibility 
> 3.1.2 바이너리 호환성

Binary code is architecture-specific. 
> 이진 코드는 아키텍처에 따라 다릅니다.

A cubin object is generated using the compiler option –code that specifies the targeted architecture: 
> cubin 객체는 타겟 아키텍처를 지정하는 컴파일러 옵션 -code를 사용하여 생성됩니다.

For example, compiling with –code=sm_13 produces binary code for devices of compute capability 1.3. 
> 예를 들어, -code=sm_13으로 컴파일하면 컴퓨팅 기능이 1.3인 디바이스에 대한 2진 코드가 생성됩니다.

Binary compatibility is guaranteed from one minor revision to the next one, but not from one minor revision to the previous one or across major revisions. 
> 바이너리 호환성은 하나의 마이너 개정에서 다음 마이너 개정으로 보장되지만, 하나의 마이너 개정에서 이전 개정으로 또는 메이저 개정 간에는 보장되지 않습니다.

In other words, a cubin object generated for compute capability X.y is only guaranteed to execute on devices of compute capability X.z where z≥y.
> 즉, 컴퓨팅 기능 X.y에 대해 생성된 cubin 객체는 z≥y가 있는 컴퓨팅 기능 X.z의 디바이스에서 실행되도록 보장됩니다.
  
3.1.3 PTX Compatibility 
> 3.1.3 PTX 호환성

Some PTX instructions are only supported on devices of higher compute capabilities. 
> 일부 PTX 명령(지침)어는 보다 높은 컴퓨팅 기능을 갖춘 디바이스에서만 지원됩니다.

For example, atomic instructions on global memory are only supported on devices of compute capability 1.1 and above; double-precision instructions are only supported on devices of compute capability 1.3 and above. 
> 예를 들어, 전역 메모리에 대한 원자적 명령어는 컴퓨팅 기능 1.1 이상의 디바이스에서만 지원됩니다. 배 정밀도 명령어는 컴퓨팅 기능이 1.3 이상인 디바이스에서만 지원됩니다.

The –arch compiler option specifies the compute capability that is assumed when compiling C to PTX code. 
> -arch 컴파일러 옵션은 C에서 PTX 코드를 컴파일할 때 가정되는 계산 기능을 지정합니다.

So, code that contains double-precision arithmetic, for example, must be compiled with “-arch=sm_13” (or higher compute capability), otherwise double-precision arithmetic will get demoted to single-precision arithmetic. 
> 예를 들어, 배정도 산술을 포함하는 코드는 "-arch=sm_13"(또는 더 높은 계산 기능)으로 컴파일해야 합니다. 그렇지 않으면 배 정밀도 산술이 단 정밀도 산술로 강등됩니다.

PTX code produced for some specific compute capability can always be compiled to binary code of greater or equal compute capability. 
> 특정 컴퓨팅 기능을 위해 생성된 PTX 코드는 언제나 더 크거나 같은 컴퓨팅 기능의 바이너리 코드로 컴파일될 수 있습니다.

## 3.1.4 Application Compatibility 
> 3.1.4 애플리케이션 호환성

To execute code on devices of specific compute capability, an application must load binary or PTX code that is compatible with this compute capability as described in Sections 3.1.2 and 3.1.3. 
> 특정 컴퓨팅 기능이 있는 디바이스에서 코드를 실행하려면 애플리케이션이 섹션 3.1.2 및 3.1.3에서 설명한 대로 이 계산 기능과 호환되는 바이너리 또는 PTX 코드를 로드해야 합니다.

In particular, to be able to execute code on future architectures with higher compute capability – for which no binary code can be generated yet –, an application must load PTX code that will be just-in-time compiled for these devices (see Section 3.1.1.2). 
> 특히 이진 코드가 아직 생성되지 않는 더 높은 컴퓨팅 기능으로 향후 아키텍처에서 코드를 실행할 수 있으려면 애플리케이션에서 이러한 디바이스용으로 just-in-time 컴파일될 PTX 코드를 로드해야 합니다 (3.1.1.2 절 참조).

Which PTX and binary code gets embedded in a CUDA C application is controlled by the –arch and –code compiler options or the –gencode compiler option as detailed in the nvcc user manual. 
> CUDA C 애플리케이션에 내장된 PTX 및 바이너리 코드는 nvcc 사용자 설명서에 자세히 설명 된 -arch 및 -code 컴파일러 옵션 또는 -gencode 컴파일러 옵션으로 제어됩니다.

For example, embeds binary code compatible with compute capability 1.0 (first –gencode option) and PTX and binary code compatible with compute capability 1.1 (second -gencode option). 
> 예를 들어 컴퓨팅 기능 1.0 (첫 번째 -gencode 옵션)과 호환되는 바이너리 코드와 PTX 및 컴퓨팅 기능 1.1 (두 번째 -gencode 옵션)과 호환되는 바이너리 코드를 포함합니다.

Host code is generated to automatically select at runtime the most appropriate code to load and execute, which, in the above example, will be:  
> 위의 예제에서 로드 및 실행에 가장 적합한 코드를 런타임에 자동으로 선택하도록 호스트 코드가 생성됩니다.

1.0 binary code for devices with compute capability 1.0,  1.1, 1.2, 1.3 binary code for devices with compute capability 1.1, 1.2, 1.3,  binary code obtained by compiling 1.1 PTX code for devices with compute capabilities 2.0 and higher. 
> 1.0 컴퓨팅 기능이 있는 디바이스용 1.0 바이너리 코드, 1.1, 1.2, 1.3 컴퓨팅 기능이 있는 디바이스용 1.1 바이너리 코드, 2.0 이상의 컴퓨팅 기능을 갖춘 디바이스용 1.1 PTX 코드를 컴파일하여 얻은 이진 코드.

x.cu can have an optimized code path that uses atomic operations, for example, which are only supported in devices of compute capability 1.1 and higher. 
> x.cu는 원자 연산을 사용하는 최적화된 코드 경로를 가질 수 있습니다. 예를 들어 컴퓨팅 기능이 1.1 이상인 디바이스에서만 지원됩니다.

The __CUDA_ARCH__ macro can be used to differentiate various code paths based on compute capability. It is only defined for device code. 
> __CUDA_ARCH__ 매크로는 계산 기능을 기반으로 다양한 코드 경로를 구별하는 데 사용할 수 있습니다. 디바이스 코드에 대해서만 정의됩니다.

When compiling with “arch=compute_11” for example, __CUDA_ARCH__ is equal to 110. 
> 예를 들어 "arch=compute_11"을 사용하여 컴파일하는 경우 __CUDA_ARCH__은 110과 동일합니다.

Applications using the driver API must compile code to separate files and explicitly load and execute the most appropriate file at runtime. 
> 드라이버 API를 사용하는 애플리케이션은 코드를 컴파일하여 파일을 분리하고 런타임에 가장 적합한 파일을 명시적으로 로드하고 실행해야 합니다.

The nvcc user manual lists various shorthands for the –arch, –code, and gencode compiler options. 
> nvcc 사용자 설명서는 -arch, -code 및 gencode 컴파일러 옵션에 대한 다양한 약어를 나열합니다.

For example, “arch=sm_13” is a shorthand for “arch=compute_13 code=compute_13,sm_13” (which is the same as “gencode arch=compute_13,code=\’compute_13,sm_13\’”). 
> 예를 들어 "arch=sm_13"은 "arch=compute_13code=compute_13,sm_13"( "gencode arch = compute_13,code = \ 'compute_13,sm_13 \'"과 동일함)의 줄임말입니다.

## 3.1.5 C/C++ Compatibility 
> 3.1.5 C/C++ 호환성

The front end of the compiler processes CUDA source files according to C++ syntax rules. Full C++ is supported for the host code. 
> 컴파일러의 프론트엔드는 C++ 구문 규칙에 따라 CUDA 소스 파일을 처리합니다. 전체 C++가 호스트 코드에 지원됩니다.

However, only a subset of C++ is fully supported for the device code as described in Appendix D. 
> 그러나 부록 D에 설명된 대로 디바이스 코드에 대해C ++의 서브세트만 전적으로 지원됩니다.

As a consequence of the use of C++ syntax rules, void pointers (e.g., returned by malloc()) cannot be assigned to non-void pointers without a typecast. 
> C++ 구문 규칙의 사용에 따라, void 포인터 (예를 들면 malloc()으로 반환됨)는 타입 변환없이 비 void 포인터에 할당될 수 없습니다.

## 3.1.6 64-Bit Compatibility 
> 3.1.6 64 비트 호환성

The 64-bit version of nvcc compiles device code in 64-bit mode (i.e. pointers are 64-bit). 
> nvcc의 64 비트 버전은 디바이스 코드를 64 비트 모드 (즉, 포인터는 64 비트)로 컴파일합니다.

Device code compiled in 64-bit mode is only supported with host code compiled in 64-bit mode. 
> 64 비트 모드로 컴파일된 디바이스 코드는 64 비트 모드로 컴파일된 호스트 코드에서만 지원됩니다.

Similarly, the 32-bit version of nvcc compiles device code in 32-bit mode and device code compiled in 32-bit mode is only supported with host code compiled in 32-bit mode. 
> 마찬가지로 nvcc의 32 비트 버전은 디바이스 코드를 32 비트 모드로 컴파일하고 디바이스 코드는 32 비트 모드로 컴파일되며 호스트 코드는 32 비트 모드로 컴파일된 경우에만 지원됩니다.

The 32-bit version of nvcc can compile device code in 64-bit mode also using the m64 compiler option. 
> nvcc의 32 비트 버전은 m64 컴파일러 옵션을 사용하여 64 비트 모드로 디바이스 코드를 컴파일할 수 있습니다.

The 64-bit version of nvcc can compile device code in 32-bit mode also using the m32 compiler option. 
> nvcc의 64 비트 버전은 m32 컴파일러 옵션을 사용하여 32 비트 모드로 디바이스 코드를 컴파일할 수 있습니다.

## 3.2 CUDA C Runtime 
> 3.2 CUDA C 런타임

The runtime is implemented in the cudart dynamic library which is typically included in the application installation package. 
> 런타임은 일반적으로 애플리케이션 설치 패키지에 포함되어 있는 큐다트(cudart) 동적 라이브러리에 구현됩니다.

All its entry points are prefixed with cuda. 
> 모든 진입점 앞에는 cuda가 붙습니다.

As mentioned in Section 2.4, the CUDA programming model assumes a system composed of a host and a device, each with their own separate memory. 
> 2.4 절에서 언급했듯이, CUDA 프로그래밍 모델은 각각 별도의 메모리를 가진 호스트와 디바이스로 구성된 시스템을 가정합니다.

Section 3.2.2 gives an overview of the runtime functions used to manage device memory. 
> 3.2.2 절은 디바이스 메모리를 관리하는 데 사용되는 런타임 기능에 대한 개요를 제공합니다.

Section 3.2.3 illustrates the use of shared memory, introduced in Section 2.2, to maximize performance. 
> 3.2.3 절은 성능을 최대화하기 위해 2.2 절에서 소개한 공유 메모리의 사용법을 보여줍니다.

Section 3.2.4 introduces page-locked host memory that is required to overlap kernel execution with data transfers between host and device memory. 
> 3.2.4 절은 호스트와 디바이스 메모리 사이의 데이터 전송으로 커널 실행을 오버랩 시키는데 필요한 페이지 잠금 호스트 메모리를 소개합니다.

Section 3.2.5 describes the concepts and API used to enable asynchronous concurrent execution at various levels in the system. 
> 3.2.5 절은 시스템의 다양한 레벨에서 비동기 동시 실행을 가능하게 하는 데 사용되는 개념과 API에 대해 설명합니다.

Section 3.2.6 shows how the programming model extends to a system with multiple devices attached to the same host. 
> 3.2.6 절은 프로그래밍 모델이 동일한 호스트에 연결된 여러 디바이스가 있는 시스템으로 확장되는 방법을 보여줍니다.

Section 3.2.8 describe how to properly check the errors generated by the runtime. 
> 3.2.8 절은 런타임으로 생성된 오류를 올바르게 검사하는 방법을 설명합니다.

Section 3.2.9 mentions the runtime functions used to manage the CUDA C call stack. 
> 3.2.9 절은 CUDA C 호출 스택을 관리하는 데 사용되는 런타임 기능에 대해 설명합니다.

Section 3.2.10 presents the texture and surface memory spaces that provide another way to access device memory; they also expose a subset of the GPU texturing hardware. 
> 3.2.10 절은 디바이스 메모리에 접근하는 또 다른 방법을 제공하는 텍스처 및 표면 메모리 공간을 표시합니다. GPU 텍스처 하드웨어의 서브셋도 노출합니다.

Section 3.2.11 introduces the various functions the runtime provides to interoperate with the two main graphics APIs, OpenGL and Direct3D. 
> 3.2.11 절은 런타임이 두 개의 메인 그래픽 API인 OpenGL과 Direct3D와 상호운용하도록 제공하는 다양한 기능을 소개합니다.

## 3.2.1 Initialization 
> 3.2.1 초기화

There is no explicit initialization function for the runtime; it initializes the first time a runtime function is called (more specifically any function other than functions from the device and version management sections of the reference manual). 
> 런타임에 대한 명시적 초기화 기능은 없습니다. 처음 런타임 함수가 호출될 때 초기화됩니다 
> (더 구체적으로는 디바이스의 기능 이외의 모든 기능과 참조 매뉴얼의 버전 관리 섹션).

One needs to keep this in mind when timing runtime function calls and when interpreting the error code from the first call into the runtime. 
> 타이밍 런타임 함수 호출할 때와 첫번째 호출에서 오류 코드를 런타임으로 해석할 때 이를 명심해야 합니다.

During initialization, the runtime creates a CUDA context for each device in the system  (see Section G.1 for more details on CUDA contexts). 
> 초기화하는 동안 런타임은 시스템의 각 디바이스에 대한 CUDA 컨텍스트를 만듭니다 (CUDA 컨텍스트에 대한 자세한 내용은 섹션 G.1 참조).

This context is the primary context for this device and it is shared among all the host threads of the application. 
> 이 컨텍스트는 이 디바이스의 기본 컨텍스트이며 애플리케이션의 모든 호스트 스레드 중에 공유됩니다.

This all happens under the hood and the runtime does not expose the primary context to the application. 
> 이것은 모두 내부에서 발생하며 런타임은 기본 컨텍스트를 애플리케이션에 노출시키지 않습니다.

When a host thread calls cudaDeviceReset(), this destroys the primary context of the device the host thread currently operates on (i.e. the current device as defined in Section 3.2.6.2). 
> 호스트 스레드가 cudaDeviceReset()를 호출하면 호스트 스레드가 현재 작동하는 디바이스 (즉, 3.2.6.2 절에 정의된 현재 디바이스)의 기본 컨텍스트를 파괴합니다.

The next runtime function call made by any host thread that has this device as current will create a new primary context for this device. 
> 현재 이 디바이스를 가지고 있는 모든 호스트 스레드로 만든 다음의 런타임 함수 호출은 이 디바이스에 대한 새로운 기본 컨텍스트를 만듭니다.

## 3.2.2 Device Memory 
> 3.2.2 디바이스 메모리

As mentioned in Section 2.4, the CUDA programming model assumes a system composed of a host and a device, each with their own separate memory. 
> 2.4 절에서 언급했듯이 CUDA 프로그래밍 모델은 각각 별도의 메모리를 가진 호스트와 디바이스로 구성된 시스템으로 가정합니다.

Kernels can only operate out of device memory, so the runtime provides functions to allocate, deallocate, and copy device memory, as well as transfer data between host memory and device memory. 
> 커널은 디바이스 메모리에서만 작동할 수 있으므로 런타임은 디바이스 메모리를 할당과 할당 해제 및 복사하는 기능을 제공하고 호스트 메모리와 디바이스 메모리 간에 데이터를 전송하는 기능도 제공합니다.

Device memory can be allocated either as linear memory or as CUDA arrays. 
> 디바이스 메모리는 선형 메모리나 CUDA 배열로 할당할 수 있습니다. 

CUDA arrays are opaque memory layouts optimized for texture fetching.
> CUDA 배열은 텍스처 가져오기에 최적화된 불투명 메모리 레이아웃입니다.

They are described in Section 3.2.10. 
> 이것들은 3.2.10 절에 설명되어 있습니다.

Linear memory exists on the device in a 32-bit address space for devices of compute capability 1.x and 40-bit address space of devices of higher compute capability, so separately allocated entities can reference one another via pointers, for example, in a binary tree. 
> 선형 메모리는 컴퓨팅 성능이 1.x인 디바이스의 경우 32 비트 주소 공간과 컴퓨팅 기능이 더 높은 디바이스의 40 비트 주소 공간의 디바이스에 있으므로, 별도로 할당된 엔티티는 포인터를 통해 서로를 참조할 수 있습니다. 예를 들면 바이너리 트리에 있습니다..

Linear memory is typically allocated using cudaMalloc() and freed using cudaFree() and data transfer between host memory and device memory are typically done using cudaMemcpy(). 
> 선형 메모리는 일반적으로 cudaMalloc()을 사용하여 할당되고 cudaFree()를 사용하여 해제되며 호스트 메모리와 디바이스 메모리 사이의 데이터 전송은 일반적으로 cudaMemcpy()를 사용하여 수행됩니다.

In the vector addition code sample of Section 2.1, the vectors need to be copied from host memory to device memory: 
> 2.1 절의 벡터 에디션 코드 샘플에서 벡터는 호스트 메모리에서 디바이스 메모리로 복사해야 합니다.

These functions are recommended for allocations of 2D or 3D arrays as it makes sure that the allocation is appropriately padded to meet the alignment requirements described in Section 5.3.2.1, therefore ensuring best performance when accessing the row addresses or performing copies between 2D arrays and other regions of device memory (using the cudaMemcpy2D() and cudaMemcpy3D() functions).
> 이러한 함수는 5.3.2.1 절에 설명된 정렬 요구 사항을 충족시키도록 할당이 적절히 패딩되도록 하므로 2D 또는 3D 배열 할당에 권장됩니다. 따라서 행 주소에 액세스하거나 2D 배열과 디바이스 메모리의 다른 영역 간에 복사본을 수행할 때 최상의 성능을 보장합니다. (cudaMemcpy2D() 및 cudaMemcpy3D() 함수를 사용하여).

The returned pitch (or stride) must be used to access array elements. 
> 반환된 피치 (또는 스트라이드)는 배열 요소에 액세스하는 데 사용해야 합니다.

The following code sample allocates a width×height 2D array of floating-point values and shows how to loop over the array elements in device code:
> 다음 코드 샘플은 부동 소수점 값의 너비×높이 2D 배열을 할당하고 디바이스 코드의 배열 요소를 반복하는 방법을 보여줍니다.

The following code sample allocates a width×height×depth 3D array of floating-point values and shows how to loop over the array elements in device code: 
> 다음 코드 샘플은 부동 소수점 값의 너비×높이×깊이 3D 배열을 할당하고 디바이스 코드의 배열 요소를 반복하는 방법을 보여줍니다.

The reference manual lists all the various functions used to copy memory between linear memory allocated with cudaMalloc(), linear memory allocated with cudaMallocPitch() or cudaMalloc3D(), CUDA arrays, and memory allocated for variables declared in global or constant memory space. 
> 참조 매뉴얼에는 cudaMalloc()으로 할당된 선형 메모리, cudaMallocPitch()나 cudaMalloc3D(), CUDA 배열로 할당된 선형 메모리, 그리고 전역 또는 상수 메모리 공간에서 선언된 변수에 할당된 메모리 간에 메모리를 복사하는 데 사용되는 다양한 함수가 모두 나열되어 있습니다.

The following code sample illustrates various ways of accessing global variables via the runtime API: 
> 다음 코드 샘플은 런타임 API를 통해 전역 변수에 액세스하는 다양한 방법을 보여줍니다.

cudaGetSymbolAddress() is used to retrieve the address pointing to the memory allocated for a variable declared in global memory space. 
> cudaGetSymbolAddress()는 전역 메모리 공간에서 선언된 변수에 할당된 메모리를 가리키는 주소를 검색하는 데 사용됩니다.

The size of the allocated memory is obtained through cudaGetSymbolSize(). 
> 할당된 메모리의 크기는 cudaGetSymbolSize()를 통해 얻습니다.

## 3.2.3 Shared Memory 
> 3.2.3 공유 메모리

As detailed in Section B.2 shared memory is allocated using the __shared__ qualifier. 
> B.2 절에서 자세히 설명했듯이 공유 메모리는 __shared__ 한정어(수식어)를 사용하여 할당됩니다. 

Shared memory is expected to be much faster than global memory as mentioned in Section 2.2 and detailed in Section 5.3.2.3. 
> 공유 메모리는 2.2 절과 5.3.2.3 절에서 자세히 언급한 것처럼 전역 메모리보다 훨씬 빨라질 것으로 예상됩니다.

Any opportunity to replace global memory accesses by shared memory accesses should therefore be exploited as illustrated by the following matrix multiplication example. 
> 따라서 다음과 같은 매트릭스 곱셈 예제에 설명된 것처럼 공유 메모리 액세스로 전역 메모리 액세스를 대체할 수 있는 모든 기회를 활용해야 합니다.

The following code sample is a straightforward implementation of matrix multiplication that does not take advantage of shared memory. 
> 다음 코드 샘플은 공유 메모리를 사용하지 않는 매트릭스 곱셈을 직접 구현한 것입니다.

Each thread reads one row of A and one column of B and computes the corresponding element of C as illustrated in Figure 3-1. 
> 각 스레드는 A의 한 행과 B의 한 열을 읽고 그림 3-1에서 설명한 대로 C의 해당 요소를 계산합니다.

A is therefore read B.width times from global memory and B is read A.height times. 
> 따라서 A는 전역 메모리에서 B.width번 읽고 B는 A.height번 읽습니다.

Figure 3-1. Matrix Multiplication without Shared Memory 
> 그림 3-1. 공유 메모리가 없는 매트릭스 곱셈

The following code sample is an implementation of matrix multiplication that does take advantage of shared memory. 
> 다음 코드 샘플은 공유 메모리를 활용하는 매트릭스 곱셈의 구현입니다.

In this implementation, each thread block is responsible for computing one square sub-matrix Csub of C and each thread within the block is responsible for computing one element of Csub. 
> 이 구현에서, 각 스레드 블록은 C의 한 사각형 서브 매트릭스 Csub를 계산하고 블록 내의 각 스레드는 Csub의 한 요소를 계산합니다.

As illustrated in Figure 3-2, Csub is equal to the product of two rectangular matrices: the sub-matrix of A of dimension (A.width, block_size) that has the same row indices as Csub, and the submatrix of B of dimension (block_size, A.width) that has the same column indices as Csub. 
> 그림 3-2에서 볼 수 있듯이 Csub는 두 개의 직사각형 매트릭스의 곱과 같습니다: Csub와 동일한 행 인덱스를 갖는 차원 (A.width, block_size)의 A의 서브 매트릭스와 차원의 B의 서브 매트릭스 (block_size, A.width)는 Csub와 같은 열 인덱스를 가집니다.

In order to fit into the device’s resources, these two rectangular matrices are divided into as many square matrices of dimension block_size as necessary and Csub is computed as the sum of the products of these square matrices. 
> 디바이스의 리소스에 맞추기 위해 이 두 개의 직사각형 매트릭스는 필요에 따라 block_size 차원의 많은 정사각형 매트릭스로 나누어지고 Csub는이 정사각형 매트릭스의 곱의 합으로 계산됩니다.

Each of these products is performed by first loading the two corresponding square matrices from global memory to shared memory with one thread loading one element of each matrix, and then by having each thread compute one element of the product. 
> 이러한 각각의 제품은 각 매트릭스의 한 요소를 로드하는 하나의 스레드로 전역 메모리에서 공유 메모리로 먼저 두 개의 대응하는 정사각형 매트릭스를 로드한 다음 각 스레드가 제품의 한 요소를 계산하도록 함으로써 수행됩니다.

Each thread accumulates the result of each of these products into a register and once done writes the result to global memory. 
> 각 스레드는 이러한 제품 각각의 결과를 레지스터에 누적하고 완료되면 결과를 전역 메모리에 작성합니다.

By blocking the computation this way, we take advantage of fast shared memory and save a lot of global memory bandwidth since A is only read (B.width / block_size) times from global memory and B is read (A.height / block_size) times. 
> 이러한 방식으로 계산을 차단함으로써, A는 전역 메모리에서  (B.width / block_size) 번 읽기 전용이고 B는 (A.height / block_size) 번 읽기 전용이기 때문에 많은 전역 메모리 대역폭을 절약합니다..

The Matrix type from the previous code sample is augmented with a stride field, so that sub-matrices can be efficiently represented with the same type. 
> 이전 코드 샘플의 매트릭스 유형에 스트라이드 필드가 추가되어 서브 매트릭스를 동일한 유형을 사용하여 효율적으로 표현할 수 있습니다.

__device__ functions (see Section B.1.1) are used to get and set elements and build any submatrix from a matrix. 
> __device__ 함수 (섹션 B.1.1 참조)는 요소를 가져와 설정하고 매트릭스에서 서브 매트릭스를  구축하는 데 사용됩니다.

## 3.2.4 Page-Locked Host Memory 
> 3.2.4 페이지 잠금 호스트 메모리

The runtime provides functions to allow the use of page-locked (also known as pinned) host memory (as opposed to regular pageable host memory allocated by malloc()):  
> 런타임은 페이지 잠금(고정으로도 알려진) 호스트 메모리 (malloc()으로 할당된 일반 페이지 가능 호스트 메모리와 반대)를 사용할 수 있는 기능을 제공합니다.

cudaHostAlloc() and cudaFreeHost() allocate and free page-locked host memory;  cudaHostRegister() page-locks a range of memory allocated by malloc() (see reference manual for limitations). 
> cudaHostAlloc()과 cudaFreeHost()는 페이지 잠금 호스트 메모리를 할당하고 해제합니다. cudaHostRegister()는 malloc()으로 할당된 메모리 범위를 페이지 잠금합니다 (제한 사항은 참조 매뉴얼 참조).

Using page-locked host memory has several benefits:  
> 페이지 잠금 호스트 메모리를 사용하면 몇 가지 이점이 있습니다.

Copies between page-locked host memory and device memory can be performed concurrently with kernel execution for some devices as mentioned in Section 3.2.5;  
> 페이지 잠금 호스트 메모리와 디바이스 메모리 사이의 복사는 3.2.5 절에서 언급한 것처럼 일부 디바이스의 커널 실행과 동시에 수행될 수 있습니다.

On some devices, page-locked host memory can be mapped into the address space of the device, eliminating the need to copy it to or from device memory as detailed in Section 3.2.4.3; 
> 어떤 디바이스에서는 페이지 잠금 호스트 메모리를 디바이스의 주소 공간에 매핑할 수 있으므로 3.2.4.3 절에서 자세히 설명한 것처럼 디바이스 메모리에서 또는 디바이스 메모리로 복사할 필요가 없습니다.

On systems with a front-side bus, bandwidth between host memory and device memory is higher if host memory is allocated as page-locked and even higher if in addition it is allocated as write-combining as described in Section 3.2.4.2. 
> 전방(프런트 사이드) 버스가 있는 시스템에서는 호스트 메모리가 페이지 잠금으로 할당된 경우 호스트 메모리와 디바이스 메모리 사이의 대역폭이 더 높으며, 3.2.4.2 절에서 설명한 대로 쓰기 조합으로 할당되는 경우 더 높습니다.

Page-locked host memory is a scarce resource however, so allocations in page-locked memory will start failing long before allocations in pageable memory. 
> 그러나 페이지 잠금 호스트 메모리는 부족한 리소스이므로 페이지 잠금 메모리의 할당은 
페이지 가능 메모리에서 할당하기 오래 전에 오류가 시작됩니다.

In addition, by reducing the amount of physical memory available to the operating system for paging, consuming too much page-locked memory reduces overall system performance. 
> 또한 페이징을 위해 운영 체제에서 사용할 수 있는 실제 메모리 양을 줄임으로써 너무 많은 페이지 잠금 메모리를 사용하면 전반적인 시스템 성능이 저하됩니다.

## 3.2.4.1 Portable Memory
> 3.2.4.1 휴대용 메모리

The simple zero-copy SDK sample comes with a detailed document on the page-locked memory APIs.
> 간단한 zero-copy SDK 샘플에는 페이지 잠금 메모리 API에 대한 자세한 문서가 있습니다.

A block of page-locked memory can be used in conjunction with any device in the system (see Section 3.2.6 for more details on multi-device systems), but by default, the benefits of using page-locked memory described above are only available in conjunction with the device that was current when the block was allocated (and with all devices sharing the same unified address space, if any, as described in Section 3.2.7). 
> 페이지 잠금 메모리 블록은 시스템의 모든 디바이스와 함께 사용할 수 있지만 (다중 디바이스  시스템에 대한 자세한 내용은 3.2.6 절을 참조하십시오) 기본적으로, 위에서 설명한 페이지 잠금 메모리를 사용하면 얻을 수 있는 이점은 블록이 할당되었을 때 현재 있던 디바이스와만 함께 사용할 수 있습니다 (모든 디바이스는 3.2.7 절에서 설명한 것처럼 동일한 통합 주소 공간을 공유합니다).

To make these advantages available to all devices, the block needs to be allocated by passing the flag cudaHostAllocPortable to cudaHostAlloc() or page-locked by passing the flag cudaHostRegisterPortable to cudaHostRegister(). 
> 이러한 장점을 모든 디바이스에 사용할 수 있도록 하려면 cudaHostRegisterPortable 플래그를 cudaHostAlloc()에 전달하여 블록을 할당하거나 cudaHostAllocPortable 플래그를 cudaHostRegister()에 전달하여 페이지 잠금을 합니다.

## 3.2.4.2 Write-Combining Memory 
> 3.2.4.2 쓰기 조합 메모리

By default page-locked host memory is allocated as cacheable. 
> 기본적으로 페이지 잠금 호스트 메모리는 캐시 가능으로 할당됩니다.

It can optionally be allocated as write-combining instead by passing flag cudaHostAllocWriteCombined to cudaHostAlloc(). 
> 선택적으로 cudaHostAllocWriteCombined 플래그를 cudaHostAlloc()에 전달하여 대신 쓰기 결합으로 할당할 수 있습니다.

Write-combining memory frees up the host’s L1 and L2 cache resources, making more cache available to the rest of the application. 
> 쓰기 조합 메모리는 호스트의 L1 및 L2 캐시 리소스를 비우고 나머지 애플리케이션에 더 많은 캐시를 사용할 수 있도록합니다.

In addition, write-combining memory is not snooped during transfers across the PCI Express bus, which can improve transfer performance by up to 40%. 
> 또한 쓰기 결합 메모리는 PCI Express 버스를 통해 전송하는 동안 스누핑되지 않으므로 전송 성능을 최대 40%까지 향상시킬 수 있습니다.

Reading from write-combining memory from the host is prohibitively slow, so write-combining memory should in general be used for memory that the host only writes to. 
> 호스트에서 쓰기 결합 메모리 읽기는 매우 느리므로 호스트가 쓰기만 하는 메모리에 일반적으로 쓰기 결합 메모리를 사용해야 합니다.

## 3.2.4.3 Mapped Memory 
> 3.2.4.3 매핑된 메모리

On devices of compute capability greater than 1.0, a block of page-locked host memory can also be mapped into the address space of the device by passing flag cudaHostAllocMapped to cudaHostAlloc() or by passing flag cudaHostRegisterMapped to cudaHostRegister(). 
> 컴퓨팅 기능이 1.0보다 큰 디바이스에서 페이지 잠금 호스트 메모리 블록은  cudaHostAllocMapped 플래그를 cudaHostAlloc()에 전달하거나 cudaHostRegisterMapped 플래그를 cudaHostRegister()에 전달하여 디바이스의 주소 공간에 매핑될 수 있습니다.

Such a block has therefore in general two addresses: one in host memory that is returned by cudaHostAlloc() or malloc(), and one in device memory that can be retrieved using cudaHostGetDevicePointer() and then used to access the block from within a kernel. 
> 따라서 이러한 블록은 일반적으로 두 개의 주소가 있습니다. 하나는 cudaHostAlloc() 또는 malloc()에 의해 반환되는 호스트 메모리와 하나는 cudaHostGetDevicePointer()를 사용하여 검색한 다음 커널 내에서 블록에 액세스하는 데 사용되는 디바이스 메모리입니다 .

The only exception is for pointers allocated with cudaHostAlloc() and when a unified address space is used for the host and the device as mentioned in Section 3.2.7.  
> 유일한 예외는 cudaHostAlloc()으로 할당된 포인터와 3.2.7 절에서 언급한 것처럼 호스트와 디바이스에 통합된 주소 공간이 사용된 경우입니다.

Accessing host memory directly from within a kernel has several advantages:  
> 커널 내에서 호스트 메모리에 직접 액세스하는 데는 몇 가지 이점이 있습니다.

There is no need to allocate a block in device memory and copy data between this block and the block in host memory; data transfers are implicitly performed as needed by the kernel; 
> 디바이스 메모리에 블록을 할당하고 이 블록과 호스트 메모리의 블록 간에 데이터를 복사할 필요가 없습니다. 데이터 전송은 커널에 의해 필요에 따라 암묵적으로 수행됩니다.
 
There is no need to use streams (see Section 3.2.5.4) to overlap data transfers with kernel execution; the kernel-originated data transfers automatically overlap with kernel execution. 
> 커널 실행으로 데이터 전송을 오버랩하기 위해 스트림 (3.2.5.4 절 참조)을 사용할 필요가 없습니다. 커널에서 시작된 데이터 전송은 커널 실행과 자동으로 겹칩니다.

Since mapped page-locked memory is shared between host and device however, the application must synchronize memory accesses using streams or events (see Section 3.2.5) to avoid any potential read-after-write, write-after-read, or write-after-write hazards. 
> 그러나 매핑된 페이지 잠금 메모리는 호스트와 디바이스 간에 공유되므로 애플리케이션은 스트림이나 이벤트 (3.2.5 절 참조)를 사용하여 메모리 액세스를 동기화해야 잠재적인 쓰기 후 읽기, 읽기 후 쓰기 또는 쓰기 후 쓰기 위험을 피해야 합니다.

To be able to retrieve the device pointer to any mapped page-locked memory, pagelocked memory mapping must be enabled by calling cudaSetDeviceFlags() with the cudaDeviceMapHost flag before any other CUDA calls is performed. 
> 매핑된 페이지 잠금 메모리에 대한 디바이스 포인터를 검색하려면 다른 CUDA 호출이 수행되기 전에 cudaDeviceMapHost 플래그가 있는 cudaSetDeviceFlags()를 호출하여 페이지 잠금 메모리 매핑을 활성화해야 합니다.

Otherwise, cudaHostGetDevicePointer() will return an error. 
> 그렇지 않으면 cudaHostGetDevicePointer()가 오류를 반환합니다.

cudaHostGetDevicePointer() also returns an error if the device does not support mapped page-locked host memory. 
> 디바이스가 매핑된 페이지 잠금 호스트 메모리를 지원하지 않으면 cudaHostGetDevicePointer() 또한 오류를 반환합니다.

Applications may query this capability by checking the canMapHostMemory device property (see Section 3.2.6.1), which is equal to 1 for devices that support mapped page-locked host memory. 
> 애플리케이션은 매핑된 페이지 잠금 호스트 메모리를 지원하는 디바이스의 경우 1과 같은
canMapHostMemory 디바이스 프로퍼티(3.2.6.1 절 참조함)를 확인하여 이 기능을 쿼리할 수 있습니다.

Note that atomic functions (Section B.11) operating on mapped page-locked memory are not atomic from the point of view of the host or other devices. 
> 매핑된 페이지 잠금 메모리에서 작동하는 원자적 기능 (섹션 B.11)은 호스트나 다른 디바이스의 관점에서 볼 때 원자적이지 않습니다.

## 3.2.5 Asynchronous Concurrent Execution 
> 3.2.5 비동기 동시 실행

## 3.2.5.1 Concurrent Execution between Host and Device 
> 3.2.5.1 호스트와 디바이스 간의 동시 실행

In order to facilitate concurrent execution between host and device, some function calls are asynchronous: 
> 호스트와 디바이스 간의 동시 실행을 용이하게 하기 위해 일부 함수 호출은 비동기식입니다.

Control is returned to the host thread before the device has completed the requested task. 
> 디바이스가 요청된 작업을 완료하기 전에 제어가 호스트 스레드로 반환됩니다.

These are:  
> 그것들은:

Kernel launches;  Memory copies between two addresses to the same device memory;  Memory copies from host to device of a memory block of 64 KB or less;  
> 커널이 시작됩니다. 두 개의 주소 사이에서 동일한 디바이스 메모리로 메모리 복사. 메모리는 64KB 이하의 메모리 블록을 호스트에서 디바이스로 복사하는 메모리.

Memory copies performed by functions that are suffixed with Async;  Memory set function calls. 
> 비동기로 끝나는 함수가 수행하는 메모리 복사. 메모리 세트 함수 호출.

Programmers can globally disable asynchronous kernel launches for all CUDA applications running on a system by setting the CUDA_LAUNCH_BLOCKING environment variable to 1. 
> 프로그래머는 CUDA_LAUNCH_BLOCKING 환경 변수를 1로 설정하여 시스템에서 실행중인 모든 CUDA 애플리케이션에 대해 비동기 커널 시작을 전역적으로 비활성화할 수 있습니다.

This feature is provided for debugging purposes only and should never be used as a way to make production software run reliably. 
> 이 기능은 디버깅 목적으로만 제공되므로 프로덕션 소프트웨어를 안정적으로 실행하는 방법으로 사용하면 안됩니다.

When an application is run via cuda-gdb, the Visual Profiler, or the Parallel Nsight CUDA Debugger, all launches are synchronous. 
> 애플리케이션이 cuda-gdb, Visual Profiler 또는 Parallel Nsight CUDA 디버거를 통해 실행될 때 모든 시작은 동기식입니다.

## 3.2.5.2 Overlap of Data Transfer and Kernel Execution 
> 3.2.5.2 데이터 전송과 커널 실행의 오버랩

Some devices of compute capability 1.1 and higher can perform copies between page-locked host memory and device memory concurrently with kernel execution. 
> 컴퓨팅 성능이 1.1 이상인 일부 디바이스는 커널 실행과 동시에 페이지 잠금 호스트 메모리와 디바이스 메모리 간에 복사를 수행할 수 있습니다.

Applications may query this capability by checking the asyncEngineCount device property (see Section 3.2.6.1), which is greater than zero for devices that support it. 
> 애플리케이션은 asyncEngineCount 디바이스 프로퍼티(3.2.6.1 절 참조)을 확인하여 이 기능을 쿼리할 수 있습니다. 이 프로퍼티는 이를 지원하는 디바이스의 경우 0보다 큽니다.

For devices of compute capability 1.x, this capability is only supported for memory copies that do not involve CUDA arrays or 2D arrays allocated through cudaMallocPitch() (see Section 3.2.2). 
> 컴퓨팅 기능 1.x인 디바이스의 경우 이 기능은 cudaMallocPitch()를 통해 할당된 CUDA 배열이나 2D 배열과 관련이 없는 메모리 복사본에 대해서만 지원됩니다 (3.2.2 절 참조).
 
## 3.2.5.3 Concurrent Kernel Execution 
> 3.2.5.3 동시 커널 실행

Some devices of compute capability 2.x and higher can execute multiple kernels concurrently. 
> 컴퓨팅 성능이 2.x 이상인 일부 디바이스는 여러 커널을 동시에 실행할 수 있습니다.

Applications may query this capability by checking the concurrentKernels device property (see Section 3.2.6.1), which is equal to 1 for devices that support it. 
> 애플리케이션은 concurrentKernels 디바이스 프로퍼티 (3.2.6.1 참조)를 확인하여 이 기능을 쿼리할 수 있습니다. 이 프로퍼티는 이를 지원하는 디바이스의 경우 1과 같습니다.

The maximum number of kernel launches that a device can execute concurrently is sixteen. 
> 디바이스가 동시에 실행할 수 있는 최대 커널 시작 수는 16입니다.

A kernel from one CUDA context cannot execute concurrently with a kernel from another CUDA context. 
> 한 CUDA 컨텍스트의 커널은 다른 CUDA 컨텍스트의 커널과 동시에 실행할 수 없습니다.

Kernels that use many textures or a large amount of local memory are less likely to execute concurrently with other kernels.  
> 많은 텍스처 또는 많은 양의 로컬 메모리를 사용하는 커널은 다른 커널과 동시에 실행될 가능성이 적습니다.

## 3.2.5.4 Concurrent Data Transfers 
> 3.2.5.4 동시 데이터 전송

Some devices of compute capability 2.x and higher can perform a copy from page-locked host memory to device memory concurrently with a copy from device memory to page-locked host memory. 
> 컴퓨팅 성능이 2.x 이상인 일부 디바이스는 페이지 잠금 호스트 메모리에서 디바이스 메모리로의 복사와 동시에 디바이스 메모리에서 페이지 잠금 호스트 메모리로 복사를 수행할 수 있습니다.

Applications may query this capability by checking the asyncEngineCount device property (see Section 3.2.6.1), which is equal to 2 for devices that support it. 
> 애플리케이션은 asyncEngineCount 디바이스 프로퍼티(3.2.6.1 절 참조)를 확인하여 이 기능을 쿼리할 수 있습니다. 이 프로퍼티는 이를 지원하는 디바이스의 경우 2와 같습니다.

## 3.2.5.5 Streams Applications manage concurrency through streams. 
> 3.2.5.5 스트림 애플리케이션은 스트림을 통해 동시성을 관리합니다.

A stream is a sequence of commands (possibly issued by different host threads) that execute in order. 
> 스트림은 순서대로 실행되는 일련의 명령입니다 (다른 호스트 스레드에서 발행되었을 수 있습니다).

Different streams, on the other hand, may execute their commands out of order with respect to one another or concurrently; this behavior is not guaranteed and should therefore not be relied upon for correctness (e.g. inter-kernel communication is undefined). 
> 한편, 다른 스트림은 서로 관련하여 또는 동시에 명령을 순서없이 실행할 수 있습니다. 이 동작은 보장되지 않으므로 정확성을 위해 의존해서는 안됩니다 (예를 들면 커널 간의 통신이 정의되지 않습니다).

## 3.2.5.5.1 Creation and Destruction 
> 3.2.5.5.1 생성과 파괴

A stream is defined by creating a stream object and specifying it as the stream parameter to a sequence of kernel launches and host device memory copies. 
> 스트림은 스트림 객체를 생성하고 커널 시작 및 호스트 디바이스 메모리 사본의 시퀀스에 대한 스트림 매개변수로 지정하여 정의됩니다.
 
The following code sample creates two streams and allocates an array hostPtr of float in page-locked memory. 
> 다음 코드 샘플은 두 개의 스트림을 만들고 페이지 잠금 메모리에 float의 hostPtr 배열을 할당합니다. 

Each stream copies its portion of input array hostPtr to array inputDevPtr in device memory, processes inputDevPtr on the device by calling MyKernel(), and copies the result outputDevPtr back to the same portion of hostPtr. 
> 각 스트림은 입력 배열 hostPtr의 일부를 디바이스 메모리의 배열 inputDevPtr에 복사하고, MyKernel()을 호출하여 디바이스의 inputDevPtr을 처리한 다음 결과 outputDevPtr을 hostPtr의 동일한 부분에 다시 복사합니다.

Section 3.2.5.5.5 describes how the streams overlap in this example depending on the capability of the device. 
> 3.2.5.5.5 절에서는 이 예에서 디바이스의 기능에 따라 스트림이 겹치는 방식을 설명합니다.

Note that hostPtr must point to page-locked host memory for any overlap to occur. 
> 오버랩이 발생하려면 hostPtr이 페이지 잠금 호스트 메모리를 가리켜야 합니다.

Streams are released by calling cudaStreamDestroy(). 
> 스트림은 cudaStreamDestroy()를 호출하여 해제됩니다.

cudaStreamDestroy() waits for all preceding commands in the given stream to complete before destroying the stream and returning control to the host thread. 
> cudaStreamDestroy()는 스트림을 파기하고 제어를 호스트 스레드로 반환하기 전에 주어진 스트림의 모든 선행 명령이 완료되기를 기다립니다.

## 3.2.5.5.2 Default Stream 
> 3.2.5.5.2 기본 스트림

Kernel launches and host device memory copies that do not specify any stream parameter, or equivalently that set the stream parameter to zero, are issued to the default stream. 
> 커널이 시작되고 스트림 매개변수를 지정하지 않거나 스트림 매개변수를 0으로 설정하는 것과 같은 호스트 메모리 복사본이 기본 스트림으로 발행됩니다.

They are therefore executed in order. 
> 따라서 이것들은 순서대로 실행됩니다.

## 3.2.5.5.3 Explicit Synchronization 
> 3.2.5.5.3 명시적 동기화

There are various ways to explicitly synchronize streams with each other. 
> 명시적으로 스트림을 서로 동기화하는 다양한 방법이 있습니다. 

cudaDeviceSynchronize() waits until all preceding commands in all streams of all host threads have completed. 
> cudaDeviceSynchronize()는 모든 호스트 스레드의 모든 스트림에 있는 모든 선행 명령이 완료될 때까지 대기합니다.

cudaStreamSynchronize() takes a stream as a parameter and waits until all preceding commands in the given stream have completed. 
> cudaStreamSynchronize()는 스트림을 매개변수로 취하고 주어진 스트림의 모든 선행 명령이 완료될 때까지 대기합니다.

It can be used to synchronize the host with a specific stream, allowing other streams to continue executing on the device. 
> 특정 스트림을 가지는 호스트를 동기화하여 다른 스트림을 디바이스에서 계속 실행할 수 있습니다.

cudaStreamWaitEvent() takes a stream and an event as parameters (see Section 3.2.5.6 for a description of events) and makes all the commands added to the given stream after the call to cudaStreamWaitEvent() delay their execution until the given event has completed. 
> cudaStreamWaitEvent()는 스트림과 이벤트를 매개변수로 취하고 (이벤트 설명은 3.2.5.6 절 참조) 주어진 이벤트가 완료될 때까지 cudaStreamWaitEvent()에 대한 호출이 실행을 지연시킨 후 모든 명령을 주어진 스트림에 추가시킵니다.

The stream can be 0, in which case all the commands added to any stream after the call to cudaStreamWaitEvent() wait on the event. 
> 스트림은 0일 수 있습니다. 이 경우 cudaStreamWaitEvent()에 대한 호출 후 스트림에 추가된 모든 명령은 이벤트를 기다립니다.

cudaStreamQuery() provides applications with a way to know if all preceding commands in a stream have completed. 
> cudaStreamQuery()는 스트림의 모든 선행 명령이 완료되었는지를 알 수 있는 방법을 애플리케이션에 제공합니다.

To avoid unnecessary slowdowns, all these synchronization functions are usually best used for timing purposes or to isolate a launch or memory copy that is failing. 
> 불필요한 속도 저하를 피하기 위해 이러한 모든 동기화 기능은 일반적으로 타이밍 목적으로 가장 잘 사용하거나 론칭과 실패한 메모리 복사본을 격리합니다.

## 3.2.5.5.4 Implicit Synchronization 
> 3.2.5.5.4 암시적 동기화

Two commands from different streams cannot run concurrently if either one of the following operations is issued in-between them by the host thread:  a page-locked host memory allocation,  a device memory allocation,  a device memory set,  a memory copy between two addresses to the same device memory,  any CUDA command to the default stream, a switch between the L1/shared memory configurations described in Section F.4.1. 
> 서로 다른 스트림의 두 명령은 호스트 스레드에 의해 다음 작업 중 하나가 실행되면 동시에 실행할 수 없습니다. 페이지 잠금 호스트 메모리 할당, 디바이스 메모리 할당, 디바이스 메모리 세트, 동일한 디바이스 메모리에 대한 두 주소 간의 메모리 복사, 기본 스트림에 대한 모든 CUDA 명령,  F.4.1 절에 설명된 L1/공유 메모리 구성 간의 전환.

For devices that support concurrent kernel execution, any operation that requires a dependency check to see if a streamed kernel launch is complete:  
> 커널 동시 실행을 지원하는 디바이스의 경우, 스트리밍된 커널 론칭이 완료되었는지 확인하기 위해 종속성 검사가 필요한 모든 작업:

Can start executing only when all thread blocks of all prior kernel launches from any stream in the CUDA context have started executing;  
> CUDA 컨텍스트의 모든 스트림에서 모든 이전 커널 론칭의 모든 스레드 블록이 실행을 시작한 경우에만 실행을 시작할 수 있습니다.

Blocks all later kernel launches from any stream in the CUDA context until the kernel launch being checked is complete. 
> 커널 론칭 점검이 완료될 때까지 CUDA 컨텍스트의 모든 스트림에서 모든 이후 커널 시작을 차단합니다.

Operations that require a dependency check include any other commands within the same stream as the launch being checked and any call to cudaStreamQuery() on that stream. 
> 종속성 점검이 필요한 작업에는 점검중인 론칭과 동일한 스트림 내의 다른 명령과 해당 스트림에서의 cudaStreamQuery() 호출이 포함됩니다.

Therefore, applications should follow these guidelines to improve their potential for concurrent kernel execution:   
> 따라서 애플리케이션은 동시 커널 실행 가능성을 높이려면 다음 지침을 따라야 합니다.

All independent operations should be issued before dependent operations, Synchronization of any kind should be delayed as long as possible. 
> 모든 독립적인 작업은 종속 작업 전에 실행되어야 하며 모든 종류의 동기화는 가능한 한 지연되어야 합니다.

## 3.2.5.5.5 Overlapping Behavior 
> 3.2.5.5.5 겹치는 동작

The amount of execution overlap between two streams depends on the order in which the commands are issued to each stream and whether or not the device supports overlap of data transfer and kernel execution (Section 3.2.5.2), concurrent kernel execution (Section 3.2.5.3), and/or concurrent data transfers (Section 3.2.5.4). 
> 두 스트림 간의 실행 오버랩 양은 각 스트림에 명령이 발행되는 순서와 디바이스가 데이터 전송 및 커널 실행 (3.2.5.2 절), 동시 커널 실행 (3.2.5.3 절 참조) 그리고/또는 동시 데이터 전송 (3.2.5.4 절)을 지원하는지 여부에 따라 다릅니다. 

For example, on devices that do not support concurrent data transfers, the two streams of the code sample of Section 3.2.5.5.1 do not overlap at all because the memory copy from host to device is issued to stream 1 after the memory copy from device to host is issued to stream 0, so it can only start once the memory copy from device to host issued to stream 0 has completed. 
> 예를 들어, 동시 데이터 전송을 지원하지 않는 디바이스의 경우, 3.2.5.5.1 절의 코드 샘플의 두 스트림은 디바이스에서 호스트로의 메모리 복사가 스트림 0에 발행된 후에, 호스트에서 디바이스로의 메모리 복사가 스트림 1에 발행되기 때문에 전혀 중복되지 않습니다. 따라서 디바이스에서 호스트로 메모리 복사가 스트림 0으로 완료된 후에만 시작할 수 있습니다.

If the code is rewritten the following way (and assuming the device supports overlap of data transfer and kernel execution) for (int i = 0; i < 2; ++i) ...  then the memory copy from host to device issued to stream 1 overlaps with the kernel launch issued to stream 0.     
> (int i = 0; i <2; ++ i)...에 대해 코드가 다음과 같은 방식으로 재작성되고 (디바이스가 데이터 전송 및 커널 실행 오버랩을 지원한다고 가정하면) 스트림 1에 발행된 호스트에서 디바이스로의 메모리 복사는 스트림 0에 발행된 커널 론칭과 오버랩됩니다.

On devices that do support concurrent data transfers, the two streams of the code sample of Section 3.2.5.5.1 do overlap: 
> 동시 데이터 전송을 지원하는 디바이스에서 섹션 3.2.5.5.1의 코드 샘플의 두 스트림이 오버랩됩니다.

The memory copy from host to device issued to stream 1 overlaps with the memory copy from device to host issued to stream 0 and even with the kernel launch issued to stream 0 (assuming the device supports overlap of data transfer and kernel execution). 
> 스트림 1로 발행된 호스트에서 디바이스로의 메모리 복사는 스트림 0으로 발행된 디바이스에서 호스트로의 메모리 복사와 오버랩되고 심지어 스트림이 0으로 발행된 커널 론칭으로도 오버랩됩니다(디바이스가 데이터 전송 및 커널 실행 오버랩을 지원한다고 가정하면).

However, the kernel executions cannot possibly overlap because the second kernel launch is issued to stream 1 after the memory copy from device to host is issued to stream 0, so it is blocked until the first kernel launch issued to stream 0 is complete as per Section 3.2.5.5.4. 
> 그러나 커널 실행은 디바이스로부터 호스트로의 메모리 복사가 스트림 0에 발행된 후, 두 번째 커널 론칭이 스트림 1에 발행되기 때문에 오버랩될 수 없습니다. 따라서 스트림 0에 발행된 첫 번째 커널 론칭이 3.2.5.5.4 절마다  완료될 때까지 차단됩니다.

If the code is rewritten as above, the kernel executions overlap (assuming the device supports concurrent kernel execution) since the second kernel launch is issued to stream 1 before the memory copy from device to host is issued to stream 0. 
> 위와 같이 코드를 다시 작성하면 디바이스에서 호스트로 메모리 복사가 스트림 0으로 발행되기 전에 두 번째 커널 론칭이 스트림 1에 발행되기 때문에 커널 실행이 겹칩니다 (디바이스가 동시 커널 실행을 지원한다고 가정함).

In that case however, the memory copy from device to host issued to stream 0 only overlaps with the last thread blocks of the kernel launch issued to stream 1 as per Section 3.2.5.5.4, which can represent only a small portion of the total execution time of the kernel. 
> 그러나 이 경우 스트림 0으로 발행된 디바이스에서 호스트로의 메모리 복사는 3.2.5.5.4 절에 따라 스트림 1에 발행된 커널 론칭의 마지막 스레드 블록과 오버랩됩니다. 이는 커널의 총 실행 시간 중 적은 부분만을 나타낼 수 있습니다.

## 3.2.5.6 Events 
> 3.2.5.6 이벤트

The runtime also provides a way to closely monitor the device’s progress, as well as perform accurate timing, by letting the application asynchronously record events at any point in the program and query when these events are completed. 
> 또한 런타임은 프로그램의 모든 지점에서 이벤트를 비동기적으로 기록하고 이러한 이벤트가 완료될 때 쿼리하도록 하여 정확한 타이밍을 수행할 뿐만 아니라 디바이스의 진행 상황을 면밀히 모니터링할 수 있는 방법을 제공합니다.

An event has completed when all tasks – or optionally, all commands in a given stream – preceding the event have completed. 
> 이벤트가 완료되기 전에 모든 작업 (또는 선택적으로 지정된 스트림의 모든 명령)이 완료되면 이벤트가 완료됩니다.

Events in stream zero are completed after all preceding task and commands in all streams are completed. 
> 스트림 0의 이벤트는 모든 선행 작업 및 모든 스트림의 명령이 완료된 후에 완료됩니다.

The following code sample creates two events: 
> 다음 코드 샘플에서는 두 가지 이벤트를 만듭니다.

They are destroyed this way: cudaEventDestroy(start); cudaEventDestroy(stop);
> 이들은 이런 식으로 망가집니다: cudaEventDestroy(start); cudaEventDestroy(stop); 

## 3.2.5.6.2 Elapsed Time 
> 경과 시간

The events created in Section 3.2.5.6.1 can be used to time the code sample of Section 3.2.5.5.1 the following way: 
> 3.2.5.6.1 절에서 생성된 이벤트는 다음과 같은 방법으로 3.2.5.5.1 절의 코드 샘플을 시간 측정하는 데 사용할 수 있습니다.

3.2.5.7 Synchronous Calls 
> 3.2.5.7 동기 호출

When a synchronous function is called, control is not returned to the host thread before the device has completed the requested task. 
> 동기 함수가 호출되면 디바이스가 요청된 태스크를 완료하기 전에 컨트롤은 호스트 스레드로 반환되지 않습니다.

Whether the host thread will then yield, block, or spin can be specified by calling cudaSetDeviceFlags() with some specific flags (see reference manual for details) before any other CUDA calls is performed by the host thread. 
> 다른 CUDA 호출이 호스트 스레드에 의해 수행되기 전에, 일부 특정 플래그와 함께 cudaSetDeviceFlags()를 호출하여 호스트 스레드가 yield, block 또는 spin 여부를 지정할 수 있습니다(자세한 내용은 참조 설명서 참조).

## 3.2.6 Multi-Device System 
> 3.2.6 다중 디바이스 시스템

## 3.2.6.1 Device Enumeration
> 3.2.6.1 디바이스 열거

A host system can have multiple devices. 
> 호스트 시스템에는 여러 장치가 있을 수 있습니다.

The following code sample shows how to enumerate these devices, query their properties, and determine the number of CUDA-enabled devices. 
> 다음 코드 샘플은 이러한 디바이스를 열거하고 해당 프로퍼티를 쿼리하고, CUDA 지원 디바이스의 수를 결정하는 방법을 보여줍니다. 

## 3.2.6.2 Device Selection 
> 3.2.6.2 디바이스 선택

A host thread can set the device it operates on at any time by calling cudaSetDevice(). 
> 호스트 스레드는 cudaSetDevice()를 호출하여 언제든지 작동하는 디바이스를 설정할 수 있습니다.

Device memory allocations and kernel launches are made on the currently set device; streams and events are created in association with the currently set device. 
> 디바이스 메모리 할당 및 커널 론칭은 현재 설정된 디바이스에서 이루어집니다. 스트림 및 이벤트는 현재 설정된 디바이스와 관련하여 생성됩니다.

If no call to cudaSetDevice() is made, the current device is device 0. 
> cudaSetDevice()에 대한 호출이 없으면, 현재 디바이스는 디바이스 0입니다.

The following code sample illustrates how setting the current device affects memory allocation and kernel execution. 
> 다음 코드 샘플은 현재 장치 설정이 메모리 할당 및 커널 실행에 미치는 영향을 보여줍니다.

A kernel launch or memory copy will fail if it is issued to a stream that is not associated to the current device as illustrated in the following code sample. 
> 다음 코드 샘플에서 설명한 것처럼 현재 디바이스와 관련이 없는 스트림에 커널 론칭 또는 메모리 복사본이 발행되면 실패합니다.

Launch kernel on device 1 in s0 cudaEventRecord() will fail if the input event and input stream are associated to different devices. 
> 입력 이벤트와 입력 스트림이 다른 디바이스와 연관되어 있으면 s0 cudaEventRecord()의 디바이스 1에서의 커널 론칭은 실패합니다. 
 
cudaEventElapsedTime() will fail if the two input events are associated to different devices. 
> 두 입력 이벤트가 다른 디바이스와 연관되어 있으면 cudaEventElapsedTime()은 실패합니다.

cudaEventSynchronize() and cudaEventQuery() will succeed even if the input event is associated to a device that is different from the current device. 
> 입력 이벤트가 현재 디바이스와 다른 디바이스와 연관되어 있더라도 cudaEventSynchronize() 및 cudaEventQuery()는 성공합니다.

cudaStreamWaitEvent() will succeed even if the input stream and input event are associated to different devices. 
> 입력 스트림과 입력 이벤트가 다른 디바이스와 연결되어 있어도 cudaStreamWaitEvent()는 성공합니다.

cudaStreamWaitEvent() can therefore be used to synchronize multiple devices with each other. 
> 따라서 cudaStreamWaitEvent()를 사용하여 여러 디바이스를 서로 동기화할 수 있습니다.

Each device has its own default stream (see Section 3.2.5.5.2), so commands issued to the default stream of a device may execute out of order or concurrently with respect to commands issued to the default stream of any other device.  
> 각 디바이스는 자체 기본 스트림 (3.2.5.5.2 절 참조)을 가지므로 디바이스의 기본 스트림에 발행된 명령은 다른 디바이스의 기본 스트림에 발행된 명령과 관련하여 순서가 없거나 동시에 실행될 수 있습니다.

## 3.2.6.4 Peer-to-Peer Memory Access 
> 3.2.6.4 피어-투-피어 메모리 액세스

When the application is run as a 64-bit process on Windows Vista/7 in TCC mode (see Section 3.6), on Windows XP, or on Linux, devices of compute capability 2.0 and higher from the Tesla series may address each other’s memory (i.e. a kernel executing on one device can dereference a pointer to the memory of the other device). 
> 애플리케이션이 TCC 모드의 Windows Vista/7 (3.6 절 참조), Windows XP 또는 리눅스에서 64 비트 프로세스로 실행되는 경우, Tesla 시리즈의 컴퓨팅 기능이 2.0 이상인 디바이스는 서로의 메모리를 주소 지정합니다 (즉, 한 디바이스에서 실행되는 커널은 다른 디바이스의 메모리에 대한 포인터를 참조 해제할 수 있습니다.

This peer-to-peer memory access feature is supported between two devices if cudaDeviceCanAccessPeer() returns true for these two devices. 
> 이 두 디바이스에 대해 cudaDeviceCanAccessPeer()가 true를 반환하면 이 피어-투-피어 메모리 액세스 기능이 두 디바이스 간에 지원됩니다.

Peer-to-peer memory access must be enabled between two devices by calling cudaDeviceEnablePeerAccess() as illustrated in the following code sample. 
> 다음 코드 샘플에서 설명하는 것처럼 cudaDeviceEnablePeerAccess()를 호출하여 두 디바이스 사이에 피어-투-피어 메모리 액세스를 사용하도록 설정해야 합니다.

A unified address space is used for both devices (see Section 3.2.7), so the same pointer can be used to address memory from both devices as shown in the code sample below. 
> 두 디바이스 모두에 통합된 주소 공간이 사용되므로 (3.2.7 절 참조) 아래 코드 샘플에서 볼 수 있듯이 두 디바이스의 메모리를 주소 지정하는 데 동일한 포인터가 사용될 수 있습니다.

This kernel launch can access memory on device 0 at address p0 
> 이 커널 론칭은 p0 주소의 디바이스 0에 있는 메모리에 액세스할 수 있습니다. 

## 3.2.6.5 Peer-to-Peer Memory Copy 
> 3.2.6.5 피어-투-피어 메모리 복사

Memory copies can be performed between the memories of two different devices. 
> 두 개의 다른 디바이스의 메모리간에 메모리 복사를 수행할 수 있습니다.

When a unified address space is used for both devices (see Section 3.2.7), this is done using the regular memory copy functions mentioned in Section 3.2.2. 
> 통합된 주소 공간이 두 디바이스 (3.2.7 절 참조)에 사용될 때, 이는 3.2.2 절에서 언급된 일반적인 메모리 복사 기능을 사용하여 수행됩니다.

Launch kernel on device 1 
> 디바이스 1에서 커널 시작

A copy between the memories of two different devices does not start until all commands previously issued to either device have completed and runs to completion before any asynchronous commands (see Section 3.2.5) issued after the copy to either device can start. 
> 서로 다른 두 디바이스의 메모리 사이의 복사는 두 디바이스 중 하나에 이전에 발행된 모든 명령이 완료될 때까지 시작되지 않고, 두 디바이스 중 하나에 대한 복사 이후에 발행된 비동기 명령 (3.2.5 절 참조)이 시작되기 전에 완료될 때까지 실행됩니다.

Note that if peer-to-peer access is enabled between two devices via cudaDeviceEnablePeerAccess() as described in Section 3.2.6.4, peer-to-peer memory copy between these two devices no longer needs to be staged through the host and is therefore faster. 
> 3.2.6.4 절에 설명된 대로 cudaDeviceEnablePeerAccess()를 통해 두 디바이스 사이에 피어-투-피어 액세스가 가능한 경우 이 두 디바이스 간의 피어-투-피어 메모리 복사는 더 이상 호스트를 통해 준비할 필요가 없으므로 더 빠릅니다 .

## 3.2.7 Unified Virtual Address Space 
> 3.2.7 통합 가상 주소 공간

For 64-bit applications on Windows Vista/7 in TCC mode (see Section 3.6), on Windows XP, or on Linux, a single address space is used for the host and all the devices of compute capability 2.0 and higher. 
> Windows Vista / 7 on TCC 모드 (3.6 절 참조), Windows XP 또는 Linux에서 64 비트 응용 프로그램의 경우 호스트 및 컴퓨팅 기능 2.0 이상의 모든 장치에 단일 주소 공간이 사용됩니다.

This address space is used for all allocations made in host memory via cudaHostAlloc() and in any of the device memories via cudaMalloc*(). 
> 이 주소 공간은 cudaHostAlloc()을 통한 호스트 메모리와 cudaMalloc*()을 통한 모든 디바이스  메모리에서 만든 모든 할당에 사용됩니다.

Which memory a pointer points to – host memory or any of the device memories – can be determined from the value of the pointer using cudaPointerGetAttributes(). 
> 포인터가 가리키는 메모리 - 호스트 메모리 또는 임의의 디바이스 메모리 -는 cudaPointerGetAttributes()를 사용하여 포인터 값에서 결정할 수 있습니다.

As a consequence:  When copying from or to the memory of one of the devices for which the unified address space is used, the cudaMemcpyKind parameter of cudaMemcpy*() becomes useless and can be set to cudaMemcpyDefault;  
> 결과로서: 통합 주소 공간이 사용되는 디바이스 중 하나의 메모리에서 또는 디바이스로 복사할 때 cudaMemcpy*()의 cudaMemcpyKind 매개변수는 쓸모없게 되고 cudaMemcpyDefault로 설정할 수 있습니다.

Allocations via cudaHostAlloc() are automatically portable (see Section 3.2.4.1) across all the devices for which the unified address space is used, and pointers returned by cudaHostAlloc() can be used directly from within kernels running on these devices (i.e. there is no need to obtain a device pointer via cudaHostGetDevicePointer() as described in Section 3.2.4.3). 
> cudaHostAlloc()을 통한 할당은 통합 주소 공간이 사용되는 모든 디바이스에서 자동으로 이동 가능하고 (3.2.4.1 절 참조) cudaHostAlloc()에서 반환된 포인터는 이러한 장치에서 실행되는 커널 내에서 직접 사용할 수 있습니다 (즉, 3.2.4.3 절에서 설명한 대로 cudaHostGetDevicePointer()를 통해 디바이스 포인터를 얻을 필요가 없습니다.

Applications may query if the unified address space is used for a particular device by checking that the unifiedAddressing device property (see Section 3.2.6.1) is equal to 1. 
> 애플리케이션은 unifiedAddressing 디바이스 프로퍼티 (3.2.6.1 절 참조)이 1과 같은지 확인하여 통합 주소 공간이 특정 디바이스에 사용되는지 여부를 쿼리할 수 있습니다.

## 3.2.8 Error Checking 
> 3.2.8 오류 검사

All runtime functions return an error code, but for an asynchronous function (see Section 3.2.5), this error code cannot possibly report any of the asynchronous errors that could occur on the device since the function returns before the device has completed the task; the error code only reports errors that occur on the host prior to executing the task, typically related to parameter validation; if an asynchronous error occurs, it will be reported by some subsequent unrelated runtime function call.  
> 모든 런타임 함수는 오류 코드를 반환하지만 비동기 함수 (3.2.5 절 참조)의 경우 이 오류 코드는 디바이스가 작업을 완료하기 전에 함수가 반환하기 때문에 디바이스에서 발생할 수 있는 비동기 오류를 보고할 수 없습니다. 오류 코드는 일반적으로 매개변수 유효성 검사와 관련된 작업을 실행하기 전에 호스트에서 발생하는 오류만 보고합니다. 비동기 오류가 발생하면 관련이 없는 런타임 함수 호출에 의해 보고됩니다.

The only way to check for asynchronous errors just after some asynchronous function call is therefore to synchronize just after the call by calling cudaDeviceSynchronize() (or by using any other synchronization mechanisms described in Section 3.2.5) and checking the error code returned by cudaDeviceSynchronize(). 
> 따라서 비동기 함수 호출 직후 비동기 오류를 검사하는 유일한 방법은 cudaDeviceSynchronize()를 호출하거나 (3.2.5 절에 설명된 다른 동기화 메커니즘을 사용하거나) 호출 직후에 동기화하고 cudaDeviceSynchronize()에서 반환한 오류 코드를 확인하는 것입니다.

The runtime maintains an error variable for each host thread that is initialized to cudaSuccess and is overwritten by the error code every time an error occurs (be it a parameter validation error or an asynchronous error). cudaPeekAtLastError() returns this variable. 
> 런타임은 cudaSuccess로 초기화되고 오류가 발생할 때마다 (매개변수 유효성 검증 오류 또는 비동기 오류 일 때) 오류 코드로 겹쳐 쓰여지는 각 호스트 스레드에 대한 오류 변수를 유지 보수합니다. cudaPeekAtLastError()는 이 변수를 반환합니다.

cudaGetLastError() returns this variable and resets it to cudaSuccess. Kernel launches do not return any error code, so cudaPeekAtLastError() or cudaGetLastError() must be called just after the kernel launch to retrieve any pre-launch errors. 
> cudaGetLastError()는 이 변수를 반환하고 cudaSuccess로 다시 설정합니다. 커널 시작은 오류 코드를 반환하지 않으므로 실행 전 오류를 검색하기 위해 커널 시작 직후에 cudaPeekAtLastError() 또는 cudaGetLastError()를 호출해야합니다.

To ensure that any error returned by cudaPeekAtLastError() or cudaGetLastError() does not originate from calls prior to the kernel launch, one has to make sure that the runtime error variable is set to cudaSuccess just before the kernel launch, for example, by calling cudaGetLastError() just before the kernel launch. 
> cudaPeekAtLastError() 또는 cudaGetLastError()가 반환한 오류가 커널 시작 전의 호출에서 비롯된 것이 아닌지 확인하려면 런타임 오류 변수가 커널 시작 직전에 cudaSuccess로 설정되어 있는지 확인해야 합니다 (예를 들어 호출 커널 시작 직전에 cudaGetLastError().

Kernel launches are asynchronous, so to check for asynchronous errors, the application must synchronize in-between the kernel launch and the call to cudaPeekAtLastError() or cudaGetLastError(). 
> 커널 시작은 비동기식이므로 비동기 오류를 검사하려면 애플리케이션이 커널 시작과 cudaPeekAtLastError() 또는 cudaGetLastError() 호출 사이에서 동기화해야 합니다.

Note that cudaErrorNotReady that may be returned by cudaStreamQuery() and cudaEventQuery() is not considered an error and is therefore not reported by cudaPeekAtLastError() or cudaGetLastError(). 
> cudaStreamQuery() 및 cudaEventQuery()에 의해 반환될 수 있는 cudaErrorNotReady는 오류로 간주되지 않으므로 cudaPeekAtLastError() 또는 cudaGetLastError()에 의해보고되지 않습니다.

## 3.2.9 Call Stack 
> 3.2.9 호출 스택

On devices of compute capability 2.x and higher, the size of the call stack can be queried using cudaDeviceGetLimit() and set using cudaDeviceSetLimit(). 
> 계산 기능이 2.x 이상인 디바이스에서 호출 스택의 크기는 cudaDeviceGetLimit()을 사용하여 쿼리하고 cudaDeviceSetLimit()을 사용하여 설정할 수 있습니다.

When the call stack overflows, the kernel call fails with a stack overflow error if the application is run via a CUDA debugger (cuda-gdb, Parallel Nsight) or an unspecified launch error, otherwise.
> 호출 스택 오버플로우가 발생하면 애플리케이션이 CUDA 디버거 (cuda-gdb, Parallel Nsight) 또는 지정되지 않은 론칭 오류를 통해 실행되면 스택 오버플로 오류로 커널 호출이 실패합니다.

## 3.2.10 Texture and Surface Memory 
> 3.2.10 텍스처 및 표면 메모리

CUDA supports a subset of the texturing hardware that the GPU uses for graphics to access texture and surface memory. 
> CUDA는 GPU가 그래픽에 사용하여 텍스처 및 표면 메모리에 액세스하는 텍스처 하드웨어의 서브세트를 지원합니다.

Reading data from texture or surface memory instead of global memory can have several performance benefits as described in Section 5.3.2.5. 
> 전역 메모리 대신에 텍스처나 표면 메모리로부터 데이터를 읽는 것은 5.3.2.5 절에서 설명한 것처럼 몇 가지 성능 이점을 가질 수 있습니다.

## 3.2.10.1 Texture Memory 
> 3.2.10.1 텍스처 메모리

Texture memory is read from kernels using the device functions described in Section B.8. 
> 텍스처 메모리는 B.8 절에서 설명한 디바이스 기능을 사용하여 커널에서 읽습니다.

The process of reading a texture is called a texture fetch. 
> 텍스처를 읽는 프로세스를 텍스처 가져오기라고 합니다.

The first parameter of a texture fetch specifies an object called a texture reference. 
> 텍스처 페치의 첫 번째 매개변수는 텍스처 참조라고하는 객체를 지정합니다.

A texture reference defines which part of texture memory is fetched. 
> 텍스처 참조는 텍스처 메모리의 어느 부분을 가져오는지를 정의합니다.

As detailed in Section 3.2.10.1.3, it must be bound through runtime functions to some region of memory, called a texture, before it can be used by a kernel. 
> 3.2.10.1.3 절에서 자세히 설명했듯이, 커널이 사용할 수 있기 전에 런타임 함수를 통해 텍스처라고 하는 메모리 영역에 바인딩해야 합니다.

Several distinct texture references might be bound to the same texture or to textures that overlap in memory. 
> 몇몇 다른 텍스처 참조는 동일한 텍스처 또는 메모리에서 겹치는 텍스처에 바인딩될 수 있습니다.
 
A texture reference has several attributes. 
> 텍스처 참조에는 여러 특성이 있습니다.

One of them is its dimensionality that specifies whether the texture is addressed as a one-dimensional array using one texture coordinate, a two-dimensional array using two texture coordinates, or a threedimensional array using three texture coordinates. 
> 그 중 하나는 텍스쳐가 하나의 텍스처 좌표를 사용하는 1차원 배열, 2개의 텍스처 좌표를 사용하는 2차원 배열 또는 3개의 텍스쳐 좌표를 사용하는 3차원 배열로 지정되는지 여부를 지정하는 차원입니다.

Elements of the array are called texels, short for “texture elements.” 
> 배열 요소는 텍셀(texels)이라고 불리며, "텍스처 요소"의 약자입니다.

The type of a texel is restricted to the basic integer and single-precision floating-point types and any of the 1-, 2-, and 4  component vector types defined in Section B.3.1 
> 텍셀의 유형은 B.3.1 절에 정의된 기본 정수 및 싱글 정밀도 부동 소수점 유형과 1, 2 및 4 구성 요소 벡터 유형으로 제한됩니다

Other attributes define the input and output data types of the texture fetch, as well as how the input coordinates are interpreted and what processing should be done. 
> 다른 특성은 텍스처 가져오기의 입력 및 출력 데이터 유형뿐만 아니라 입력 좌표가 해석되는 방식과 처리가 수행되어야 하는 방식을 정의합니다.

A texture can be any region of linear memory or a CUDA array (described in Section 3.2.10.2.3). 
> 텍스처는 선형 메모리 또는 CUDA 배열의 모든 영역이 될 수 있습니다 (3.2.10.2.3 절에서 설명).

Table F-2 lists the maximum texture width, height, and depth depending on the compute capability of the device. Textures can also be layered as described in Section 3.2.10.1.5. 
> 표 F-2는 디바이스의 계산 기능에 따라 최대 텍스처 폭, 높이 및 깊이를 나열합니다. 텍스처는 3.2.10.1.5 절에 설명된 대로 계층화할 수도 있습니다.

## 3.2.10.1.1 Texture Reference Declaration 
> 3.2.10.1.1 텍스처 참조 선언

Some of the attributes of a texture reference are immutable and must be known at compile time; they are specified when declaring the texture reference. 
> 텍스처 참조의 속성 중 일부는 변경할 수 없으므로 컴파일 시 알려야 합니다. 텍스처 참조를 선언할 때 지정됩니다.

A texture reference is declared at file scope as a variable of type texture: 
> 텍스처 참조는 파일 범위에서 텍스처 유형의 변수로 선언됩니다.

DataType specifies the type of data that is returned when fetching the texture;
> DataType은 텍스처를 가져올 때 반환되는 데이터 유형을 지정합니다.

Type is restricted to the basic integer and single-precision floating-point types and any of the 1-, 2-, and 4-component vector types defined in Section B.3.1;  
> 유형은 B.3.1 절에 정의된 기본 정수 및 싱글 정밀도 부동 소수점 유형과 1, 2 및 4 구성 요소 벡터 유형으로 제한됩니다.

Type specifies the type of the texture reference and is equal to cudaTextureType1D, cudaTextureType2D, or cudaTextureType3D, for a one-dimensional, two-dimensional, or three-dimensional texture, respectively, or cudaTextureType1DLayered or cudaTextureType2DLayered for a one-dimensional or two-dimensional layered texture respectively; 
> 타입은 1차원, 2차원 또는 3차원 텍스처 각각에 대해 cudaTextureType1D, cudaTextureType2D 또는 cudaTextureType3D와 동일하거나 1 차원 또는 2 차원 텍스처에 대해 cudaTextureType1DLayered 또는 cudaTextureType2DLayered 계층화된 텍스처 각각에 대해 텍스처 참조 유형을 지정합니다.

Type is an optional argument which defaults to cudaTextureType1D;  
> 타입은 기본적으로 cudaTextureType1D 옵션 변수입니다.

ReadMode is equal to cudaReadModeNormalizedFloat or cudaReadModeElementType; 
> ReadMode는 cudaReadModeNormalizedFloat 또는 cudaReadModeElementType과 같습니다.

if it is cudaReadModeNormalizedFloat and Type is a 16-bit or 8-bit integer type, the value is actually returned as floating-point type and the full range of the integer type is mapped to [0.0, 1.0] for unsigned integer type and [-1.0, 1.0] for signed integer type; for example, an unsigned 8-bit texture element with the value 0xff reads as 1; 
> cudaReadModeNormalizedFloat이고 타입이 16 비트 또는 8 비트 정수 유형인 경우, 이 값은 실제로 부동 소수점 유형으로 반환되고 정수 유형의 전체 범위는 부호 없는 정수 유형의 경우 [0.0, 1.0]에 매핑되고 부호 있는 정수 유형의 경우 [ -1.0, 1.0]에 매핑됩니다; 예를 들어 값 0xff를 갖는 부호 없는 8 비트 텍스처 요소는 1로 읽습니다.

if it is cudaReadModeElementType, no conversion is performed; 
> cudaReadModeElementType인 경우 변환이 수행되지 않습니다.

ReadMode is an optional argument which defaults to cudaReadModeElementType. 
> ReadMode는 기본값인 cudaReadModeElementType의 선택적 변수입니다.

A texture reference can only be declared as a static global variable and cannot be passed as an argument to a function. 
> 텍스처 참조는 정적 전역 변수로만 선언할 수 있으며 함수에 변수로 전달할 수 없습니다.

## 3.2.10.1.2 Runtime Texture Reference Attributes 
> 3.2.10.1.2 런타임 텍스처 참조 특성

The other attributes of a texture reference are mutable and can be changed at runtime through the host runtime. 
> 텍스처 참조의 다른 특성은 변경 가능하며 호스트 런타임을 통해 런타임에서 변경할 수 있습니다.

They specify whether texture coordinates are normalized or not, the addressing mode, and texture filtering, as detailed below. 
> 아래에 설명된 대로 텍스처 좌표가 정규화되었는지 여부, 주소 지정 모드 및 텍스처 필터링을 지정합니다.
 
By default, textures are referenced (by the functions of Section B.8) using floating point coordinates in the range [0, N-1] where N is the size of the texture in the dimension corresponding to the coordinate. 
> 기본적으로 텍스처는 N이 좌표에 해당하는 차원의 텍스처 크기에서 [0, N-1] 범위의 부동 소수점 좌표를 사용하여 섹션 B.8의 기능에 의해 참조됩니다. 

For example, a texture that is 6432 in size will be referenced with coordinates in the range [0, 63] and [0, 31] for the x and y dimensions, respectively. 
> 예를 들어 크기가 6432인 텍스처는 각각 x 및 y 차원에 대해 [0, 63] 및 [0, 31] 범위의 좌표로 참조됩니다.

Normalized texture coordinates cause the coordinates to be specified in the range [0.0, 1.0-1/N] instead of [0, N-1], so the same 6432 texture would be addressed by normalized coordinates in the range [0, 1-1/N] in both the x and y dimensions. 
> 표준화된 텍스처 좌표는 좌표가 [0, N-1] 대신 [0.0, 1.0-1/N] 범위에 지정되므로 동일한 6432 텍스처는 x와 y 차원에서 [0, 1-1/N] 범위의 정규화된 좌표에 의해 주소 지정됩니다.

Normalized texture coordinates are a natural fit to some applications’ requirements, if it is preferable for the texture coordinates to be independent of the texture size. 
> 정규화된 텍스처 좌표는 텍스처 좌표가 텍스처 크기와 독립적인 것이 바람직한 경우 일부 애플리케이션의 요구사항에 자연스럽게 부합합니다.

It is valid to call the device functions of Section B.8 with coordinates that are be out of range. 
> 범위를 벗어난 좌표로 섹션 B.8의 디바이스 기능을 호출하는 것은 유효합니다.

The addressing mode defines what happens if that case. 
> 주소 지정 모드는 이 경우 어떨지를 정의합니다.

The default addressing mode is to clamp the coordinates to the valid range: [0, N) for nonnormalized coordinates and [0.0, 1.0) for normalized coordinates. 
> 기본 주소 지정 모드는 좌표를 유효한 범위로 고정하는 것입니다. 즉, 비표준 좌표의 경우 [0, N]이고 정규화된 좌표의 경우 [0.0, 1.0]입니다.

If the border mode is specified instead, texture fetches with out-of-range texture coordinates return zero. 
> 대신 경계 모드가 지정되면, 범위를 벗어난 텍스처 좌표로 텍스처를 가져오면 0을 반환합니다.

For normalized coordinates, the warp mode and the mirror mode are also available. 
> 정규화된 좌표의 경우 워프 모드와 미러 모드도 사용할 수 있습니다.

When using the wrap mode, each coordinate x is converted to frac(x)=x-floor(x. where floor(x) is the largest integer not greater than x. 
> 랩 모드를 사용할 때 각 좌표 x는 frac(x)=x-floor(x, floor(x)는 x보다 크지 않은 가장 큰 정수가 있는) 입니다.

When using the mirror mode, each coordinate x is converted to frac(x) if floor(x) is even and 1-frac(x) if floor(x) is odd. 
> 미러 모드를 사용할 때 각 좌표 x는 floor(x)가 짝수일 경우 frac (x)로 변환되고 floor(x)가 홀수인 경우 1-frac(x)로 변환됩니다.

Linear texture filtering may be done only for textures that are configured to return floating-point data. 
> 선형 텍스처 필터링은 부동 소수점 데이터를 반환하도록 구성된 텍스처에 대해서만 수행할 수 있습니다. 

It performs low-precision interpolation between neighboring texels. 
> 인접한 텍셀간에 고정밀 보간을 수행합니다.

When enabled, the texels surrounding a texture fetch location are read and the return value of the texture fetch is interpolated based on where the texture coordinates fell between the texels. 
> 사용 설정되면 텍스처 가져오기 위치를 둘러싼 텍셀이 읽히고 텍스처 가져오기의 반환값은 텍스처 좌표가 텍셀 사이에서 떨어지는 위치에 기초하여 보간합니다.

Simple linear interpolation is performed for one-dimensional textures, bilinear interpolation for two-dimensional textures, and trilinear interpolation for three-dimensional textures. 
> 단순 선형 보간은 일차원 텍스처에 대해서는 단순 선형 보간이 수행되고, 2차원 텍스처에 대해서는 쌍 선형 보간이 수행되고, 3차원 텍스처에 대해서는 삼선형 보간이 수행됩니다.

Appendix E gives more details on texture fetching. 
> 부록 E에서는 텍스처 가져오기에 대해 자세히 설명합니다.

## 3.2.10.1.3 Texture Binding
> 3.2.10.1.3 텍스처 바인딩
 
As explained in the reference manual, the runtime API has a low-level C-style interface and a high-level C++-style interface. 
> 참조 설명서에서 설명한 것처럼 런타임 API에는 저수준 C 스타일 인터페이스와 고급 C++ 스타일 인터페이스가 있습니다.

The texture type is defined in the high-level API as a structure publicly derived from the textureReference type defined in the low-level API as such:  normalized specifies whether texture coordinates are normalized or not, as described in Section 3.2.10.1.2;  
> 텍스처 유형은 하위 수준 API에서 정의된 textureReference 유형에서 공개적으로 파생된 구조같은 상위 수준 API에서 정의됩니다. normalized는 3.2.10.1.2 절에서 설명한 대로 텍스처 좌표가 정규화되는지 여부를 지정합니다.

filterMode specifies the filtering mode, that is how the value returned when fetching the texture is computed based on the input texture coordinates; 
> filterMode는 필터링 모드를 지정합니다. 즉, 텍스처를 가져올 때 입력 텍스처 좌표를 기반으로 값이 반환되는 방식입니다.

filterMode is equal to cudaFilterModePoint or cudaFilterModeLinear; 
> filterMode는 cudaFilterModePoint 또는 cudaFilterModeLinear와 같습니다.

if it is cudaFilterModePoint, the returned value is the texel whose texture coordinates are the closest to the input texture coordinates; 
> cudaFilterModePoint인 경우 반환값은 텍스처 좌표가 입력 텍스처 좌표에 가장 가까운 텍셀입니다.

if it is cudaFilterModeLinear, the returned value is the linear interpolation of the two (for a one-dimensional texture), four (for a two-dimensional texture), or eight (for a three-dimensional texture) texels whose texture coordinates are the closest to the input texture coordinates; 
> cudaFilterModeLinear인 경우 반환값은 두 개의 (1 차원 텍스처의 경우), 4 개의 (2 차원 텍스처의 경우) 또는 8 개의 (3 차원 텍스처의 경우) 텍스처 좌표가 입력 텍스처 좌표에 가장 가까운 선형 보간입니다.

cudaFilterModeLinear is only valid for returned values of floating-point type;  addressMode specifies the addressing mode, as described in Section 3.2.10.1.2; 
> cudaFilterModeLinear는 부동 소수점 유형의 반환값에만 유효합니다. addressMode는 3.2.10.1.2 절에서 설명한 주소 지정 모드를 지정합니다.

addressMode is an array of size three whose first, second, and third elements specify the addressing mode for the first, second, and third texture coordinates, respectively; 
> addressMode는 제1, 제2, 및 제3 요소가 각각 제1, 제2, 및 제3의 텍스처 좌표의 주소 지정 모드를 지정하는 사이즈 3의 배열입니다.

the addressing mode are cudaAddressModeBorder, cudaAddressModeClamp, cudaAddressModeWrap, and cudaAddressModeMirror; 
> 주소 지정 모드는 cudaAddressModeBorder, cudaAddressModeClamp, cudaAddressModeWrap 및 cudaAddressModeMirror입니다.

cudaAddressModeWrap and cudaAddressModeMirror are only supported for normalized texture coordinates;  
> cudaAddressModeWrap 및 cudaAddressModeMirror는 정규화된 텍스처 좌표에만 지원됩니다.

channelDesc describes the format of the value that is returned when fetching the texture; channelDesc is of the following type: 
> channelDesc는 텍스처를 가져올 때 반환되는 값의 형식을 설명합니다. channelDesc는 다음과 같은 유형입니다.

where x, y, z, and w are equal to the number of bits of each component of the returned value and f is:  
> x, y, z 및 w는 반환값의 각 구성 요소 비트 수와 같고 f는 다음과 같습니다.

cudaChannelFormatKindSigned if these components are of signed integer type,  cudaChannelFormatKindUnsigned if they are of unsigned integer type,  cudaChannelFormatKindFloat if they are of floating point type. 
normalized, addressMode, and filterMode may be directly modified in host code. 
> 이러한 구성 요소가 부호있는 정수 유형인 경우 cudaChannelFormatKind, 부호없는 정수 유형인 경우 cudaChannelFormatKindUnsigned, 부동 소수점 유형인 경우 cudaChannelFormatKindUnsigned, normalized, addressMode 및 filterMode는 호스트 코드에서 직접 수정할 수 있습니다.

Before a kernel can use a texture reference to read from texture memory, the texture reference must be bound to a texture using cudaBindTexture() or cudaBindTexture2D() for linear memory, or cudaBindTextureToArray() for CUDA arrays. 
> 커널이 텍스처 참조를 사용하여 텍스처 메모리에서 읽기 전에, 선형 메모리의 경우 cudaBindTexture() 또는 cudaBindTexture2D()를 사용하거나 CUDA 배열의 경우 cudaBindTextureToArray()를 사용하여 텍스처 참조는 텍스처에 바인딩해야 합니다. 

cudaUnbindTexture() is used to unbind a texture reference. 
> cudaUnbindTexture()는 텍스처 참조를 바인딩 해제하는 데 사용됩니다.

It is recommended to allocate two-dimensional textures in linear memory using cudaMallocPitch() and use the pitch returned by cudaMallocPitch() as input parameter to cudaBindTexture2D().  
> cudaMallocPitch()를 사용하여 선형 메모리에 2 차원 텍스처를 할당하고 cudaMallocPitch()가 반환하는 피치를 cudaBindTexture2D()의 입력 매개변수로 사용하는 것이 좋습니다.

The following code samples bind a texture reference to linear memory pointed to by devPtr:  
> 다음 코드 샘플은 devPtr이 가리키는 선형 메모리에 텍스처 참조를 바인딩합니다.

Using the high-level API: texture<float, cudaTextureType2D, 
> 하이 레벨 API 사용: texture<float, cudaTextureType2D, 

The following code samples bind a texture reference to a CUDA array cuArray:   
> 다음 코드 샘플은 텍스처 참조를 CUDA 배열인 cuArray에 바인딩합니다.

The format specified when binding a texture to a texture reference must match the parameters specified when declaring the texture reference; otherwise, the results of texture fetches are undefined. 
> 텍스처를 텍스처 참조에 바인딩할 때 지정된 형식은 텍스처 참조를 선언할 때 지정된 매개변수와 일치해야 합니다. 그렇지 않으면 텍스처 가져오기의 결과가 정의되지 않습니다.

The following code sample applies some simple transformation kernel to a texture. 
> 다음 코드 샘플은 간단한 변환 커널을 텍스처에 적용합니다.

## 3.2.10.1.4 16-Bit Floating-Point Textures 
> 3.2.10.1.4 16 비트 부동 소수점 텍스처

The 16-bit floating-point or half format supported by CUDA arrays is the same as the IEEE 754-2008 binary2 format. 
> CUDA 배열이 지원하는 16 비트 부동 소수점 또는 반 포맷은 IEEE 754-2008 바이너리2 포맷과 동일합니다.

CUDA C does not support a matching data type, but provides intrinsic functions to convert to and from the 32-bit floating-point format via the unsigned short type: __float2half_rn(float) and __half2float(unsigned short). 
> CUDA C는 일치하는 데이터 유형을 지원하지 않지만 __float2half_rn (float) 및 __half2float (unsigned short)와 같은 부호없는 short 유형을 통해 32 비트 부동 소수점 형식과의 변환을 위한 내장 함수를 제공합니다.

These functions are only supported in device code. 
> 이러한 기능은 디바이스 코드에서만 지원됩니다.

Equivalent functions for the host code can be found in the OpenEXR library, for example. 
> 호스트 코드에 해당하는 기능은 OpenEXR 라이브러리에서 찾을 수 있습니다.

16-bit floating-point components are promoted to 32 bit float during texture fetching before any filtering is performed. 
> 필터링을 수행하기 전에 텍스처를 가져오는 동안 16 비트 부동 소수점 구성 요소가 32 비트 부동 소수점으로 승격됩니다.

A channel description for the 16-bit floating-point format can be created by calling one of the cudaCreateChannelDescHalf*() functions. 
> 16 비트 부동 소수점 형식에 대한 채널 설명은 cudaCreateChannelDescHalf*() 함수 중 하나를 호출하여 만들 수 있습니다.
 
## 3.2.10.1.5 Layered Textures 
> 3.2.10.1.5 계층화된 텍스처

A one-dimensional or two-dimensional layered texture (also know as texture array in Direct3D and array texture in OpenGL) is a texture made up of a sequence of layers, all of which are regular textures of same dimensionality, size, and data type.  
> 1 차원 또는 2 차원의 계층화된 텍스처 (Direct3D의 텍스처 배열 및 OpenGL의 배열 텍스처라고도 함)는 일련의 레이어로 구성된 텍스처로, 이 모두는 동일한 차원, 크기 및 데이터 유형의 규칙적인 텍스처입니다.

A one-dimensional layered texture is addressed using an integer index and a floating-point texture coordinate; the index denotes a layer within the sequence and the coordinate addresses a texel within that layer. 
> 1 차원 계층화된 텍스처는 정수 인덱스와 부동 소수점 텍스처 좌표를 사용하여 처리됩니다. 인덱스는 시퀀스 내의 레이어를 나타내며 좌표는 해당 레이어 내의 텍셀을 처리합니다.

A two-dimensional layered texture is addressed using an integer index and two floating-point texture coordinates; the index denotes a layer within the sequence and the coordinates address a texel within that layer. 
> 2 차원 계층화된 텍스처는 정수 인덱스와 두 개의 부동 소수점 텍스처 좌표를 사용하여 주소 지정됩니다. 인덱스는 시퀀스 내의 레이어를 나타내며 좌표는 해당 레이어 내의 텍셀을 주소 지정합니다.

A layered texture can only be bound to a CUDA array created by calling cudaMalloc3DArray() with the cudaArrayLayered flag (and a height of zero for one-dimensional layered texture). 
> 계층화된 텍스처는 cudaArrayLayered 플래그 (1 차원의 계층화된 텍스처의 높이 0인)와 함께 cudaMalloc3DArray()를 호출하여 생성된 CUDA 배열에만 바인딩할 수 있습니다.

Layered textures are fetched using the device functions described in Sections B.8.5 and B.8.6. 
> 계층화된 텍스처는 B.8.5 절과 B.8.6 절에 설명된 디바이스 기능을 사용하여 가져옵니다.

Texture filtering (see Appendix E) is done only within a layer, not across layers. 
> 텍스처 필터링 (부록 E 참조)은 계층 내에서만 수행되고 계층 간에는 수행되지 않습니다.

Layered textures are only supported on devices of compute capability 2.0 and higher. 
> 계층화된 텍스처는 컴퓨팅 기능이 2.0 이상인 디바이스에서만 지원됩니다.

## 3.2.10.1.6 Cubemap Textures 
> 3.2.10.1.6 큐브맵 텍스처

A cubemap texture is a special type of two-dimensional layered texture that has six layers representing the faces of a cube:  
> 큐브맵 텍스처는 큐브의 면을 나타내는 6 개의 계층을 가진 특별한 유형의 2 차원 계층화된 텍스처입니다.

The width of a layer is equal to its height.  
> 계층의 너비는 높이와 같습니다.

The cubemap is addressed using three texture coordinates x, y, and z that are interpreted as a direction vector emanating from the center of the cube and pointing to one face of the cube and a texel within the layer corresponding to that face. 
> 큐브맵은 3 개의 텍스처 좌표 x, y 및 z를 사용하여 지정됩니다. x, y 및 z는 큐브의 중심에서 나오는 방향 벡터로 해석되며 큐브의 한 면과 해당 면에 해당하는 계층 내의 텍셀을 가리 킵니다.

More specifically, the face is selected by the coordinate with largest magnitude m and the corresponding layer is addressed using coordinates (s/m+1)/2 and (t/m+1)/2 where s and t are defined in Table 3-1. 
> 보다 구체적으로, 면은 가장 큰 크기 m을 갖는 좌표에 의해 선택되고 표 3-1에 정의된 s와 t가 있는 좌표 (s/m+1)/2와 (t/m+1)/2를 사용하여 해당 계층이 처리됩니다.

A layered texture can only be bound to a CUDA array created by calling cudaMalloc3DArray() with the cudaArrayCubemap flag. 
> 계층화된 텍스처는 cudaArrayCubemap 플래그로 cudaMalloc3DArray()를 호출하여 생성된 CUDA 배열에만 바인딩될 수 있습니다.

Cubemap textures are fetched using the device function described in Sections B.8.7. 
> 큐브맵 텍스처는 B.8.7 절에서 설명한 디바이스 함수를 사용하여 가져옵니다.

Cubemap textures are only supported on devices of compute capability 2.0 and higher. 
> 큐브맵 텍스처는 컴퓨팅 기능이 2.0 이상인 디바이스에서만 지원됩니다.
 
## 3.2.10.1.7 Cubemap Layered Textures 
> 3.2.10.1.7 큐브맵 계층화된 텍스처

A cubemap layered texture is a layered texture whose layers are cubemaps of same dimension. 
> 큐브맵 계층화된 텍스처는 레이어가 동일한 차원의 큐브맵인 계층화된 텍스처입니다.

A cubemap layered texture is addressed using an integer index and three floating-point texture coordinates; the index denotes a cubemap within the sequence and the coordinates address a texel within that cubemap. 
> 큐브 계층화된 텍스처는 정수 인덱스와 세 개의 부동 소수점 텍스처 좌표를 사용하여 주소 지정됩니다. 인덱스는 시퀀스 내의 큐브맵을 나타내고 좌표는 큐브맵 내의 텍셀을 주소 지정합니다.

A layered texture can only be bound to a CUDA array created by calling cudaMalloc3DArray() with the cudaArrayLayered and cudaArrayCubemap flags. 
> 계층화된 텍스처는 cudaArrayLayered 및 cudaArrayCubemap 플래그를 사용하여 cudaMalloc3DArray()를 호출하여 생성된 CUDA 배열에만 바인딩할 수 있습니다.

Cubemap layered textures are fetched using the device function described in Sections B.8.8. 
> 큐브맵 계층화된 텍스처는 B.8.8 절에 설명된 디바이스 기능을 사용하여 가져옵니다.

Texture filtering (see Appendix E) is done only within a layer, not across layers. 
> 텍스처 필터링 (부록 E 참조)은 계층 내에서만 수행되며 계층 간에는 수행되지 않습니다.

Cubemap layered textures are only supported on devices of compute capability 2.0 and higher. 
> Cubemap 계층화된 텍스처는 컴퓨팅 기능이 2.0 이상인 디바이스에서만 지원됩니다.

## 3.2.10.1.8 Texture Gather 
> 3.2.10.1.8 텍스처 수집

Texture gather is a special texture fetch that is available for two-dimensional textures only. 
> 텍스처 집합은 2 차원 텍스처에만 사용할 수 있는 특수한 텍스처 가져오기입니다.

It is performed by the tex2Dgather() function, which has the same parameters as tex2D(), plus an additional comp parameter equal to 0, 1, 2, or 3 (see Section B.8.9). 
> 이것은 0, 1, 2이나 3과 같은 추가 comp 매개변수에 더하여 tex2D() 함수와 동일한 매개변수를 갖는 tex2Dgather() 함수에 의해 수행됩니다.(B.8.9 절 참조).

It returns four 32-bit numbers that correspond to the value of the component comp of each of the four texels that would have been used for bilinear filtering during a regular texture fetch. 
> 정규 텍스처 가져오기 중에 바이리니어(쌍일차) 필터링에 사용된 4 개의 텍셀 각각의 컴포넌트 comp 값에 해당하는 4 개의 32 비트 숫자를 반환합니다.

For example, if these texels are of values (253, 20, 31, 255), (250, 25, 29, 254), (249, 16, 37, 253), (251, 22, 30, 250), and comp is 2, tex2Dgather() returns (31, 29, 37, 30). 
> 예를 들어, 이들 텍셀들이 (253, 20, 31, 255), (250,25,29,254), (249,16,37,253), (251,22,30,250) comp가 2이면 tex2Dgather()가 (31, 29, 37, 30)을 반환합니다.

Texture gather is only supported for CUDA arrays created with the cudaArrayTextureGather flag and of width and height less than the maximum specified in Table F-2 for texture gather, which is smaller than for regular texture fetch. 
> 텍스처 수집은 cudaArrayTextureGather 플래그로 생성되고 텍스처 수집을 위해 표 F-2에 지정된 최대값보다 작은 너비와 높이의 CUDA 배열에 대해서만 지원되며 일반 텍스처 가져오기보다는 작습니다.

Texture gather is only supported on devices of compute capability 2.0 and higher. 
> 텍스처 수집은 컴퓨팅 기능이 2.0 이상인 디바이스에서만 지원됩니다.

## 3.2.10.2 Surface Memory 
> 3.2.10.2 표면 메모리

For devices of compute capability 2.0 and higher, a CUDA array (described in Section 3.2.10.2.3), created with the cudaArraySurfaceLoadStore flag, can be read and written via a surface reference using the functions described in Section B.9. 
> 컴퓨팅 기능이 2.0 이상인 디바이스의 경우, cudaArraySurfaceLoadStore 플래그로 생성된 CUDA 배열 (3.2.10.2.3 절에서 설명한)은 B.9 절에서 설명한 함수를 사용하여 표면 참조를 통해 읽고 쓸 수 있습니다.

Table F-2 lists the maximum surface width, height, and depth depending on the compute capability of the device. 
> 표 F-2는 디바이스의 컴퓨팅 기능에 따라 최대 표면 폭, 높이 및 깊이를 나열합니다.

3.2.10.2.1 Surface Reference Declaration 
> 3.2.10.2.1 표면 참조 선언

A surface reference is declared at file scope as a variable of type surface: 
where Type specifies the type of the surface reference and is equal to cudaSurfaceType1D, cudaSurfaceType2D, cudaSurfaceType3D,  cudaSurfaceTypeCubemap, cudaSurfaceType1DLayered, cudaSurfaceType2DLayered, or cudaSurfaceTypeCubemapLayered; 
> 표면 참조는 파일 범위에서 표면 유형의 변수로 선언됩니다. Type이 표면 참조의 유형을 지정한 데에서 cudaSurfaceType1D, cudaSurfaceType2D, cudaSurfaceType3D, cudaSurfaceTypeCubemap, cudaSurfaceType1DLayered, cudaSurfaceType2DLayered 또는 cudaSurfaceTypeCubemapLayered와 같습니다.

Type is an optional argument which defaults to cudaSurfaceType1D. 
> Type은 기본적으로 cudaSurfaceType1D 옵션 변수입니다.
 
A surface reference can only be declared as a static global variable and cannot be passed as an argument to a function. 
> 표면 참조는 정적 전역 변수로만 선언할 수 있으며 함수에 변수로 전달할 수 없습니다.

## 3.2.10.2.2 Surface Binding 
> 3.2.10.2.2 표면 바인딩(결합)

Before a kernel can use a surface reference to access a CUDA array, the surface reference must be bound to the CUDA array using cudaBindSurfaceToArray(). 
> 커널이 표면 참조를 사용하여 CUDA 배열에 액세스하기 전에 cudaBindSurfaceToArray()를 사용하여 표면 참조를 CUDA 배열에 바인딩해야 합니다.

The following code samples bind a surface reference to a CUDA array cuArray:   
> 다음 코드 샘플은 표면 참조를 CUDA 배열 cuArray에 바인딩합니다.

A CUDA array must be read and written using surface functions of matching dimensionality and type and via a surface reference of matching dimensionality; otherwise, the results of reading and writing the CUDA array are undefined. 
> CUDA 배열은 차원과 유형이 일치하는 표면 기능을 사용하고 일치하는 차원의 표면 참조를 통하여 읽고 쓰여야 합니다. 그렇지 않으면 CUDA 배열 읽기 및 쓰기 결과가 정의되지 않습니다.

Unlike texture memory, surface memory uses byte addressing. 
> 텍스처 메모리와 달리 표면 메모리는 바이트 주소 지정을 사용합니다.

This means that the x-coordinate used to access a texture element via texture functions needs to be multiplied by the byte size of the element to access the same element via a surface function. 
> 즉, 텍스처 함수를 통해 텍스처 요소에 액세스하는 데 사용되는 x 좌표에 표면 함수를 통해 동일한 요소에 액세스하려면 요소의 바이트 크기를 곱해야 합니다.

For example, the element at texture coordinate x of a one-dimensional floating-point CUDA array bound to a texture reference texRef and a surface reference surfRef is read using tex1d(texRef, x) via texRef, but surf1Dread(surfRef, 4*x) via surfRef. 
> 예를 들어 텍스처 참조 texRef와 표면 참조 surfRef에 바인딩된 1 차원 부동 소수점 CUDA 배열의 텍스처 좌표 x에 있는 요소는 surfRef를 통한 surf1Dread(surfRef, 4*x)가 아닌 texRef를 통한 tex1d(texRef, x)를 사용하여 읽습니다.

Similarly, the element at texture coordinate x and y of a two-dimensional floating-point CUDA array bound to a texture reference texRef and a surface reference surfRef is accessed using tex2d(texRef, x, y) via texRef, but surf2Dread(surfRef, 4*x, y) via surfRef (the byte offset of the y-coordinate is internally calculated from the underlying line pitch of the CUDA array). 
> 유사하게 텍스처 참조 texRef와 표면 참조 surfRef에 바인딩된 2 차원 부동 소수점 CUDA 배열의 텍스처 좌표 x와 y의 요소는 texRef를 통해 tex2d (texRef, x, y)를 사용하여 액세스되지만 surfRef를 통해 (Y 좌표의 바이트 오프셋은 내부적으로 CUDA 배열의 기본 피치로부터 계산됩니다) surf2Dread(surfRef, 4 * x, y)를 사용하지는 않습니다. .

The following code sample applies some simple transformation kernel to a texture. 
> 다음 코드 샘플은 간단한 변환 커널을 텍스처에 적용합니다.

## 3.2.10.2.3 Cubemap Surfaces 
> 3.2.10.2.3 큐브맵 표면

Cubemap surfaces are accessed using surfCubemapread() and surfCubemapwrite() (Sections B.9.11 and B.9.12) as a two-dimensional layered surface, i.e. using an integer index denoting a face and two floating-point texture coordinates addressing a texel within the layer corresponding to this face. 
> 큐브맵 표면은 surfCubemapread() 및 surfCubemapwrite() (B.9.11 절 및 B.9.12 절)를 사용하여 2 차원 계층화된 표면으로 액세스됩니다. 즉, 면을 나타내는 정수 인덱스와 두 개의 부동 소수점 텍스처 좌표를 사용하여 이 면에 해당하는 계층 내의 텍셀을 주소 지정합니다.

Faces are ordered as indicated in Table 3-1. 
> 면은 표 3-1에 표시된 대로 정렬됩니다.

## 3.2.10.2.4 Cubemap Layered Surfaces 
> 3.2.10.2.4 큐브맵 계층화된 표면

Cubemap layered surfaces are accessed using surfCubemapLayeredread() and surfCubemapLayeredwrite() (Sections B.9.13 and B.9.14) as a twodimensional layered surface, i.e. using an integer index denoting a face of one of the cubemaps and two floating-point texture coordinates addressing a texel within the layer corresponding to this face. 
> 큐브맵 계층화된 표면은 surfCubemapLayeredread() 및 surfCubemapLayeredwrite() (섹션 B.9.13 및 B.9.14)를 사용하여 2 차원 계층화된 표면으로 액세스됩니다. 즉, 큐브맵 중 하나의 면을 나타내는 정수 인덱스와 두 개의 부동 소수점 텍스처 좌표를 사용하여 이 면에 해당하는 레이어 내의 텍셀을 주소 지정합니다.

Faces are ordered as indicated in Table 3-1, so  index ((2 * 6) + 3), for example, accesses the fourth face of the third cubemap. 
> 면은 표 3-1에 표시된 대로 정렬되므로, 예를 들어 index ((2 * 6) + 3)는 세 번째 큐브맵의 네 번째 면에 액세스합니다.

## 3.2.10.3 CUDA Arrays 
> 3.2.10.3 CUDA 배열

CUDA arrays are opaque memory layouts optimized for texture fetching. 
> CUDA 배열은 텍스처 가져오기에 최적화된 불투명 메모리 레이아웃입니다.

They are one-dimensional, two-dimensional, or three-dimensional and composed of elements, each of which has 1, 2 or 4 components that may be signed or unsigned 8-, 16- or 32-bit integers, 16-bit floats, or 32-bit floats. 
> 그것들은 1 차원, 2 차원 또는 3 차원이며 각각 1, 2 또는 4 개의 구성 요소가 있는 요소로 구성됩니다. 부호가 있거나 부호가 없는 8, 16 또는 32 비트 정수, 16 비트 부동 소수점 또는 32 비트 부동 소수점 수일 수  있습니다.

CUDA arrays are only readable by kernels through texture fetching and may only be bound to texture references with the same number of packed components. 
> CUDA 배열은 텍스처 가져오기를 통해 커널에서만 읽을 수 있으며 동일한 수의 압축된 구성 요소가 있는 텍스처 참조에만 바인딩될 수 있습니다.

## 3.2.10.4 Read/Write Coherency 
> 3.2.10.4 읽기/쓰기 일관성

The texture and surface memory is cached (see Section 5.3.2.5) and within the same kernel call, the cache is not kept coherent with respect to global memory writes and surface memory writes, so any texture fetch or surface read to an address that has been written to via a global write or a surface write in the same kernel call returns undefined data. 
> 텍스처와 표면 메모리가 캐쉬되어 (5.3.2.5 절 참조) 같은 커널 호출 내에서 전역 메모리 쓰기와 표면 메모리 쓰기와 관련하여 캐시가 일관성있게 유지되지 않으므로 텍스처 가져오기나 표면 읽기가 동일한 커널 호출에서 전역 쓰기 또는 표면 쓰기를 통해 쓰여지면 정의되지 않은 데이터를 반환합니다.

In other words, a thread can safely read some texture or surface memory location only if this memory location has been updated by a previous kernel call or memory copy, but not if it has been previously updated by the same thread or another thread from the same kernel call. 
> 바꾸어 말하면, 스레드는 이전의 커널 호출이나 메모리 복사에 의해 이 메모리 위치가 업데이트 되었을 경우에만, 일부의 텍스처나 표면 메모리의 위치를 안전하게 읽어낼 수가 있습니다. 하지만 동일한 스레드나 동일한 커널 호출의 다른 스레드에 의해 이전에 업데이트된 경우는 없습니다.

## 3.2.11 Graphics Interoperability 
> 3.2.11 그래픽 상호운용

Some resources from OpenGL and Direct3D may be mapped into the address space of CUDA, either to enable CUDA to read data written by OpenGL or Direct3D, or to enable CUDA to write data for consumption by OpenGL or Direct3D. 
> OpenGL 및 Direct3D의 일부 리소스는 CUDA의 주소 공간에 매핑될 수 있으며,
CUDA가 OpenGL 또는 Direct3D로 작성된 데이터를 읽거나 CUDA가 OpenGL 또는 Direct3D에서 소비할 데이터를 쓸 수 있게 할 수 있습니다.

A resource must be registered to CUDA before it can be mapped using the functions mentioned in Sections 3.2.11.1 and 3.2.11.2. 
> 리소스는 3.2.11.1 및 3.2.11.2 절에서 언급한 기능을 사용하여 매핑될 수 있기 전에 CUDA에 등록되어야 합니다.

These functions return a pointer to a CUDA graphics resource of type struct cudaGraphicsResource. 
> 이러한 함수는 struct cudaGraphicsResource 유형의 CUDA 그래픽 리소스에 대한 포인터를 반환합니다.

Registering a resource is potentially high-overhead and therefore typically called only once per resource. 
> 리소스를 등록하는 것은 잠재적으로 높은 오버헤드이므로 일반적으로 자원 당 한 번만 호출됩니다.

A CUDA graphics resource is unregistered using cudaGraphicsUnregisterResource(). 
> CUDA 그래픽 리소스는 cudaGraphicsUnregisterResource()를 사용하여 등록이 해제됩니다. 

Once a resource is registered to CUDA, it can be mapped and unmapped as many times as necessary using cudaGraphicsMapResources() and cudaGraphicsUnmapResources(). 
> 리소스가 CUDA에 등록되면 cudaGraphicsMapResources() 및 cudaGraphicsUnmapResources()를 사용하여 필요한 만큼 매핑 및 매핑 해제할 수 있습니다.

cudaGraphicsResourceSetMapFlags() can be called to specify usage hints (write-only, read-only) that the CUDA driver can use to optimize resource management. 
> cudaGraphicsResourceSetMapFlags()를 호출하여 CUDA 드라이버가 리소스 관리를 최적화하는 데 사용할 수 있는 사용 힌트 (쓰기 전용, 읽기 전용)를 지정할 수 있습니다.

A mapped resource can be read from or written to by kernels using the device memory address returned by cudaGraphicsResourceGetMappedPointer() for buffers and cudaGraphicsSubResourceGetMappedArray() for CUDA arrays. 
> 매핑된 리소스는 버퍼에 대해 cudaGraphicsResourceGetMappedPointer() 및 CUDA 배열에 대해 cudaGraphicsSubResourceGetMappedArray()가 반환하는 디바이스 메모리 주소를 사용하여 커널에서 읽고 쓸 수 있습니다.

Accessing a resource through OpenGL or Direct3D while it is mapped to CUDA produces undefined results. 
> OpenGL 또는 Direct3D를 통해 CUDA에 매핑되는 동안 리소스에 액세스하면 정의되지 않은 결과가 발생합니다.

Sections 3.2.11.1 and 3.2.11.2 give specifics for each graphics API and some code samples. 
> 3.2.11.1 절과 3.2.11.2 절은 각 그래픽 API 및 몇 가지 코드 샘플에 대한 세부 사항을 제공합니다.

Section 3.2.11.3 gives specifics for when the system is in SLI mode. 
> 3.2.11.3 절은 시스템이 SLI 모드에있을 때의 세부 사항을 제공합니다.
 
3.2.11.1 OpenGL Interoperability 
> 3.2.11.1 OpenGL 상호운용

Interoperability with OpenGL requires that the CUDA device be specified by cudaGLSetGLDevice() before any other runtime calls. 
> OpenGL과의 상호운용을 위해서는 다른 런타임 호출보다 먼저 cudaGLSetGLDevice()가 CUDA 디바이스를 지정해야 합니다.

Note that cudaSetDevice()and cudaGLSetGLDevice() are mutually exclusive. 
> cudaSetDevice()와 cudaGLSetGLDevice()는 상호배타적임을 유의하십시오.

The OpenGL resources that may be mapped into the address space of CUDA are OpenGL buffer, texture, and renderbuffer objects. 
> CUDA의 주소 공간에 매핑될 수 있는 OpenGL 리소스는 OpenGL 버퍼, 텍스처 및 렌더버퍼 객체입니다.

A buffer object is registered using cudaGraphicsGLRegisterBuffer(). 
> 버퍼 객체는 cudaGraphicsGLRegisterBuffer()를 사용하여 등록됩니다.

In CUDA, it appears as a device pointer and can therefore be read and written by kernels or via cudaMemcpy() calls. 
> CUDA에서는 디바이스 포인터로 나타나므로 커널이나 cudaMemcpy() 호출을 통해 읽고 쓸 수 있습니다.

A texture or renderbuffer object is registered using cudaGraphicsGLRegisterImage(). In CUDA, it appears as a CUDA array. 
> 텍스처 또는 렌더버퍼 객체는 cudaGraphicsGLRegisterImage()를 사용하여 등록됩니다. CUDA에서는 CUDA 배열로 나타납니다.

Kernels can read from the array by binding it to a texture or surface reference. 
> 커널은 텍스처 또는 표면 참조에 바인딩하여 배열에서 읽을 수 있습니다.

They can also write to it via the surface write functions if the resource has been registered with the cudaGraphicsRegisterFlagsSurfaceLoadStore flag.  
> 리소스가 cudaGraphicsRegisterFlagsSurfaceLoadStore 플래그로 등록된 경우 표면 쓰기 기능을 통해 리소스에 쓸 수 있습니다.

The array can also be read and written via cudaMemcpy2D() calls. 
> 배열은 cudaMemcpy2D() 호출을 통해 읽고 쓸 수 있습니다.

cudaGraphicsGLRegisterImage() supports all texture formats with 1, 2, or 4 components and an internal type of float (e.g. GL_RGBA_FLOAT32), normalized integer (e.g. GL_RGBA8, GL_INTENSITY16), and unnormalized integer (e.g. GL_RGBA8UI) (please note that since unnormalized integer formats require OpenGL 3.0, they can only be written by shaders, not the fixed function pipeline). 
> cudaGraphicsGLRegisterImage()는 1, 2 또는 4 구성 요소와 float의 내부 유형 (예를 들어 GL_RGBA_FLOAT32), 정규화 정수 (예를 들어 GL_RGBA8, GL_INTENSITY16) 및 정규화되지 않은 정수 (예를 들어 GL_RGBA8UI)를 포함한 모든 텍스처 형식을 지원합니다. (비정규화된 정수 형식은 OpenGL 3.0이 필요하기 때문에 고정 함수 파이프라인이 아닌 쉐이더에서만 쓸 수 있다는 점에 유의하십시오).

The OpenGL context whose resources are being shared has to be current to the host thread making any OpenGL interoperability API calls. 
> 리소스가 공유되는 OpenGL 컨텍스트는 OpenGL 상호운용 API 호출을 수행하는 호스트 스레드에 최신 상태여야 합니다.

The following code sample uses a kernel to dynamically modify a 2D width x height grid of vertices stored in a vertex buffer object: 
> 다음 코드 샘플은 커널을 사용하여 정점 버퍼 객체에 저장된 정점의 2D 폭 x 높이 그리드를 동적으로 수정합니다.

On Windows and for Quadro GPUs, cudaWGLGetDevice() can be used to retrieve the CUDA device associated to the handle returned by wglEnumGpusNV(). 
> Windows 및 Quadro GPU의 경우 cudaWGLGetDevice()를 사용하여 wglEnumGpusNV()가 반환된 핸들과 연결된 CUDA 디바이스를 검색할 수 있습니다.

Quadro GPUs offer higher performance OpenGL interoperability than GeForce and Tesla GPUs in a multi-GPU configuration where OpenGL rendering is performed on the Quadro GPU and CUDA computations are performed on other GPUs in the system.  
> Quadro GPU는 OpenGL 렌더링이 Quadro GPU에서 수행되고 CUDA 계산이 시스템의 다른 GPU에서 수행되는 다중 GPU 구성에서 GeForce 및 Tesla GPU보다 높은 성능의 OpenGL 상호운용을 제공합니다.

## 3.2.11.2 Direct3D Interoperability 
> 3.2.11.2 Direct3D 상호운용

Direct3D interoperability is supported for Direct3D 9, Direct3D 10, and Direct3D 11. 
> Direct3D 상호운용은 Direct3D 9, Direct3D 10 및 Direct3D 11에서 지원됩니다.

A CUDA context may interoperate with only one Direct3D device at a time and the CUDA context and Direct3D device must be created on the same GPU. 
> CUDA 컨텍스트는 한 번에 하나의 Direct3D 디바이스와만 상호운용될 수 있으며 CUDA 컨텍스트와 Direct3D 디바이스는 동일한 GPU에서 만들어야 합니다.

In addition the following considerations must be taken when creating the device: 
> 또한 디바이스를 만들 때 다음 사항을 고려해야 합니다.

Direct3D 9 devices must be created with DeviceType set to D3DDEVTYPE_HAL and BehaviorFlags with the D3DCREATE_HARDWARE_VERTEXPROCESSING flag. 
> DeviceType을 D3DDEVTYPE_HAL로 설정하고 BehaviorFlags를 D3DCREATE_HARDWARE_VERTEXPROCESSING 플래그로 설정하여 Direct3D 9 디바이스를 만들어야 합니다.

Direct3D 10 and Direct3D 11 devices must be created with DriverType set to D3D_DRIVER_TYPE_HARDWARE. 
> DriverType을 D3D_DRIVER_TYPE_HARDWARE로 설정하여 Direct3D 10 및 Direct3D 11 디바이스를 만들어야 합니다.

Interoperability with Direct3D requires that the Direct3D device be specified by cudaD3D9SetDirect3DDevice(), cudaD3D10SetDirect3DDevice() and cudaD3D11SetDirect3DDevice(), before any other runtime calls. 
> Direct3D와의 상호운용을 위해서는 Direct3D 디바이스가 다른 런타임 호출보다 먼저 cudaD3D9SetDirect3DDevice(), cudaD3D10SetDirect3DDevice() 및 cudaD3D11SetDirect3DDevice()에 의해 지정되어야 합니다.

cudaD3D9GetDevice(), cudaD3D10GetDevice(), and cudaD3D11GetDevice() can be used to retrieve the CUDA device associated to some adapter. 
> cudaD3D9SetDirect3DDevice(), cudaD3D10SetDirect3DDevice() 및 cudaD3D11SetDirect3DDevice()에 의해 지정되어야 합니다.

A set of calls is also available to allow the creation of CUDA contexts with interoperability with Direct3D devices that use NVIDIA SLI in AFR (Alternate Frame Rendering) mode: cudaD3D[9|10|11]GetDevices(). 
> AFR (Alternate Frame Rendering) 모드에서 NVIDIA SLI를 사용하는 Direct3D 디바이스와의 상호운용을 갖춘 CUDA 컨텍스트 생성을 허용하는 호출 세트도 사용할 수 있습니다. cudaD3D [9|10|11]GetDevices().

A call to cudaD3D[9|10|11]GetDevices()can be used to obtain a list of CUDA device handles that can be passed as the (optional) last parameter to cudaD3D[9|10|11]SetDirect3DDevice(). 
> cudaD3D[9|10|11] GetDevices()에 대한 호출은 cudaD3D[9|10|11]SetDirect3DDevice()에 마지막 매개변수 (선택적)로 전달될 수 있는 CUDA 디바이스 핸들 목록을 얻는 데 사용할 수 있습니다.

The application has the choice to either create multiple CPU threads, each using a different CUDA context, or a single CPU thread using multiple CUDA context. 
> 애플리케이션은 여러 CUDA 컨텍스트를 사용하는 여러 CPU 스레드를 만들거나 여러 CUDA 컨텍스트를 사용하는 단일 CPU 스레드를 만들 수 있습니다.

If using separate CPU threads for each GPU each of the CUDA contexts would be created by the CUDA runtime by calling in a separate CPU thread cudaD3D[9|10|11]SetDirect3DDevice() using one of the CUDA device handles returned by cudaD3D[9|10|11]GetDevices(). 
> 각 GPU에 별도의 CPU 스레드를 사용하는 경우 각 CUDA 컨텍스트는 cudaD3D[9|10|11]GetDevices()가 반환된 CUDA 디바이스 핸들 중 하나를 사용하여 별도의 CPU 스레드 cudaD3D[9|10|11]SetDirect3DDevice()에서 호출하여 CUDA 런타임에서 생성됩니다 

If using a single CPU thread the CUDA contexts would have to be created using the CUDA driver API context creation functions for interoperability with Direct3D devices that use NVIDIA SLI (cuD3D[9|10|11]CtxCreateOnDevice()). 
> 단일 CPU 스레드를 사용하는 경우 NVIDIA SLI (cuD3D[9|10|11]CtxCreateOnDevice())를 사용하는 Direct3D 디바이스와의 상호운용을 위한 CUDA 드라이버 API 컨텍스트 생성 기능을 사용하여 CUDA 컨텍스트를 만들어야 합니다.

The application relies on the interoperability between CUDA driver and runtime APIs (Section G.4), which allows it to call cuCtxPushCurrent() and cuCtxPopCurrent()to change the CUDA context active at a given time. 
> 이 애플리케이션은 cuCtxPushCurrent() 및 cuCtxPopCurrent()를 호출하여 주어진 시간에 CUDA 컨텍스트를 활성화하도록 허용하는 CUDA 드라이버와 런타임 API (G.4 절) 간의 상호운용에 의존합니다.

The Direct3D resources that may be mapped into the address space of CUDA are Direct3D buffers, textures, and surfaces. 
> CUDA의 주소 공간에 매핑될 수있는 Direct3D 리소스는 Direct3D 버퍼, 텍스처 및 표면입니다.

These resources are registered using cudaGraphicsD3D9RegisterResource(), 
cudaGraphicsD3D10RegisterResource(), and cudaGraphicsD3D11RegisterResource(). 
> 이러한 리소스는 cudaGraphicsD3D9RegisterResource(), 
cudaGraphicsD3D10RegisterResource() 및 cudaGraphicsD3D11RegisterResource()를 사용하여 등록됩니다.

The following code sample uses a kernel to dynamically modify a 2D width x height grid of vertices stored in a vertex buffer object. 
> 다음 코드 샘플은 커널을 사용하여 정점 버퍼 객체에 저장된 정점의 2D 너비 x 높이 그리드를 동적으로 수정합니다. 
 
## 3.2.11.3 SLI Interoperability 
> 3.2.11.3 SLI 상호운용

In a system with multiple GPUs, all CUDA-enabled GPUs are accessible via the CUDA driver and runtime as separate devices. 
> 여러 GPU가 있는 시스템에서 모든 CUDA 지원 GPU는 CUDA 드라이버와 런타임을 통해 별도의 디바이스로 액세스할 수 있습니다.

There are however special considerations as described below when the system is in SLI mode. 
> 그러나 시스템이 SLI 모드에 있을 때 아래에 설명된 것과 같이 특별한 고려사항이 있습니다.

First, an allocation in one CUDA device on one GPU will consume memory on other GPUs that are part of the SLI configuration of the Direct3D or OpenGL device. 
> 첫째, 하나의 GPU에서 하나의 CUDA 디바이스에 할당하면 Direct3D 또는 OpenGL 디바이스의 SLI 구성의 일부인 다른 GPU의 메모리가 소모됩니다.

Because of this, allocations may fail earlier than otherwise expected. 
> 이 때문에 할당은 예상보다 빨리 실패할 수 있습니다.

Second, applications have to create multiple CUDA contexts, one for each GPU in the SLI configuration and deal with the fact that a different GPU is used for rendering by the Direct3D or OpenGL device at every frame. 
> 둘째, 애플리케이션은 SLI 구성의 각 GPU에 하나씩, 다중 CUDA 컨텍스트를 만들어야 하며 모든 프레임에서 Direct3D 또는 OpenGL 디바이스로 렌더링할 때 다른 GPU가 사용된다는 사실을 다뤄야 합니다.

The application can use the cudaD3D[9|10|11]GetDevices() for Direct3D and cudaGLGetDevices() for OpenGL set of calls to identify the CUDA device handle(s) for the device(s) that are performing the rendering in the current and next frame. 
> 애플리케이션은 Direct3D의 경우 cudaD3D[9|10l11]GetDevices()를 사용하고 OpenGL 호출 세트의 경우 cudaGLGetDevices()를 사용하여 현재 및 다음 프레임의 렌더링을 수행하고 있는 디바이스의 CUDA 디바이스 핸들을 식별할 수 있습니다.

Given this information the application will typically map Direct3D or OpenGL resources to the CUDA context corresponding to the CUDA device returned by cudaD3D[9|10|11]GetDevices() or cudaGLGetDevices() when the deviceList parameter is set to CU_D3D10_DEVICE_LIST_CURRENT_FRAME or cudaGLDeviceListCurrentFrame. 
> 이 정보가 주어지면 애플리케이션은 일반적으로 Direct3D 또는 OpenGL 리소스를 deviceList 매개변수가 CU_D3D10_DEVICE_LIST_CURRENT_FRAME 또는 cudaGLDeviceListCurrentFrame으로 설정될 때 cudaD3D[9|10|11]GetDevices()나 cudaGLGetDevices()로 반환된 CUDA 디바이스에 해당하는 CUDA 컨텍스트에 매핑합니다.

See Sections 3.2.11.2 and 3.2.11.1 for details on how the CUDA runtime interoperate with Direct3D and OpenGL, respectively. 
> CUDA 런타임이 Direct3D 및 OpenGL과 각각 어떻게 상호운용되는지에 대한 자세한 내용은 3.2.11.2 및 3.2.11.1 절을 참조하십시오.

## 3.3 Versioning and Compatibility 
> 3.3 버전 관리 및 호환성

There are two version numbers that developers should care about when developing a CUDA application: 
> 개발자가 CUDA 애플리케이션을 개발할 때 주의해야 할 두 가지 버전 번호가 있습니다.

The compute capability that describes the general specifications and features of the compute device (see Section 2.5) and the version of the CUDA driver API that describes the features supported by the driver API and runtime. 
> 일반 명세 및 컴퓨팅 디바이스(2.5 절 참조)와 기능을 설명하는 컴퓨팅 기능과 드라이버 API 및 런타임에서 지원된 기능을 설명하는 CUDA 드라이버 API 버전을 설명하는 컴퓨팅 기능이 있습니다.

The version of the driver API is defined in the driver header file as CUDA_VERSION. 
> 드라이버 API의 버전은 드라이버 헤더 파일에서 CUDA_VERSION으로 정의됩니다.

It allows developers to check whether their application requires a newer device driver than the one currently installed. 
> 개발자는 자신의 애플리케이션이 현재 설치된 디바이스보다 더 새로운 디바이스 드라이버가 필요한지 여부를 확인할 수 있습니다.

This is important, because the driver API is backward compatible, meaning that applications, plug-ins, and libraries (including the C runtime) compiled against a particular version of the driver API will continue to work on subsequent device driver releases as illustrated in Figure 3-3. 
> 이는 드라이버 API가 이전 버전과 호환되므로 즉, 즉, 드라이버 API의 특정 버전에 대해 컴파일된 애플리케이션, 플러그인 및 라이브러리 (C 런타임 포함)가 그림 3-3에서 설명한 대로 후속 디바이스 드라이버 릴리스에서 계속 동작한다는 의미로 중요합니다..

The driver API is not forward compatible, which means that applications, plug-ins, and libraries (including the C runtime) compiled against a particular version of the driver API will not work on previous versions of the device driver. 
> 드라이버 API는 포워드(순방향)와 호환되지 않습니다. 즉, 드라이버 API의 특정 버전에 대해 컴파일된 애플리케이션, 플러그인 및 라이브러리 (C 런타임 포함)가 디바이스 드라이버의 이전 버전에서 작동하지 않습니다.

It is important to note that mixing and matching versions is not supported; specifically:  
> 믹싱 및 매칭 버전은 지원되지 않는다는 것을 유의하십시오. 구체적으로:

All applications, plug-ins, and libraries on a system must use the same version of the CUDA driver API, since only one version of the CUDA device driver can be installed on a system.  
> 이는 CUDA 디바이스 드라이버의 한 버전만 시스템에 설치할 수 있기 때문에, 시스템의 모든 애플리케이션, 플러그인 및 라이브러리는 동일한 버전의 CUDA 드라이버 API를 사용해야 합니다. 

All plug-ins and libraries used by an application must use the same version of the runtime. 
> 애플리케이션에서 사용하는 모든 플러그인과 라이브러리는 동일한 버전의 런타임을 사용해야 합니다. 

All plug-ins and libraries used by an application must use the same version of any libraries that use the runtime (such as CUFFT, CUBLAS, …). 
> 애플리케이션에서 사용하는 모든 플러그인 및 라이브러리는 런타임 (예 : CUFFT, CUBLAS ...)을 사용하는 동일한 버전의 모든 라이브러리를 사용해야 합니다. 

Figure 3-3. The Driver API is Backward, but Not Forward Compatible 
> 그림 3-3. 드라이버 API는 역방향이지만 포워드(순방향)와 호환되지 않습니다.

## 3.4 Compute Modes 
> 3.4 계산 모드

On Tesla solutions running Windows Server 2008 and later or Linux, one can set any device in a system in one of the three following modes using NVIDIA’s System Management Interface (nvidia-smi), which is a tool distributed as part of the driver:  
> Windows Server 2008 이상 및 리눅스를 실행하는 Tesla 솔루션에서 드라이버의 일부로 배포되는 도구인 NVIDIA의 시스템 관리 인터페이스 (nvidia-smi)를 사용하여 다음 3 가지 모드 중 하나로 시스템의 모든 디바이스를 설정할 수 있습니다.

Default compute mode: Multiple host threads can use the device (by calling cudaSetDevice() on this device, when using the runtime API, or by making current a context associated to the device, when using the driver API) at the same time.  
> 기본 컴퓨팅 모드: 다중 호스트 스레드가 런타임 API를 사용할 때 이 디바이스에서 cudaSetDevice()를 호출하거나 동시에 드라이버 API를 사용할 때 현재 컨텍스트를 디바이스와 연관시킴으로써) 디바이스를 사용할 수 있습니다.

Exclusive-process compute mode: Only one CUDA context may be created on the device across all processes in the system and that context may be current to as many threads as desired within the process that created that context.  
> 독점 프로세스 계산 모드: 시스템의 모든 프로세스에 걸쳐 디바이스에 하나의 CUDA 컨텍스트만 생성될 수 있으며, 해당 컨텍스트를 생성한 프로세스 내에서 원하는 수의 스레드에 대해 최신일 수 있습니다.

Exclusive-process-and-thread compute mode: Only one CUDA context may be created on the device across all processes in the system and that context may only be current to one thread at a time.  
> 독점 프로세스 및 스레드 계산 모드: 시스템의 모든 프로세스에서 디바이스에 하나의 CUDA 컨텍스트만 생성될 수 있으며 해당 컨텍스트는 한 번에 하나의 스레드에만 최신일 수 있습니다.

Prohibited compute mode: No CUDA context can be created on the device. 
> 금지된 컴퓨팅 모드: 디바이스에 CUDA 컨텍스트를 만들 수 없습니다.

This means, in particular, that a host thread using the runtime API without explicitly calling cudaSetDevice() might be associated with a device other than device 0 if device 0 turns out to be in the exclusive-process mode and used by another process, or in the exclusive-process-and-thread mode and used by another thread, or in prohibited mode.
> 이것은 특히 디바이스 0이 배타적 프로세스 모드인 것으로 판명되고 다른 프로세서에 사용되거나 배타적 프로세스 및 스레드 모드에서 사용되고 다른 스레드에서 사용되거나 금지된 모드인 것으로 판명되면 cudaSetDevice()를 명시적으로 호출하지 않고 런타임 API를 사용하는 호스트 스레드가 디바이스 0 이외의 디바이스와 연관될 수 있음을 의미합니다

cudaSetValidDevices() can be used to set a device from a prioritized list of devices. 
> cudaSetValidDevices()는 우선 순위가 매겨진 디바이스 목록에서 디바이스를 설정하는 데 사용할 수 있습니다.

Applications may query the compute mode of a device by checking the computeMode device property (see Section 3.2.6.1). 
> 애플리케이션은 computeMode 디바이스 프로퍼티 (3.2.6.1 절 참조)를 확인하여 디바이스의 컴퓨팅 모드를 쿼리할 수 있습니다.
 
## 3.5 Mode Switches 
> 3.5 모드 스위치

GPUs that have a display output dedicate some DRAM memory to the so-called primary surface, which is used to refresh the display device whose output is viewed by the user. 
> 디스플레이 출력을 갖는 GPU는 사용자가 볼 수 있는 디스플레이 디바이스를 새로고침하는 데 사용되는 소위 기본 표면에 일부 DRAM 메모리를 제공합니다.

When users initiate a mode switch of the display by changing the resolution or bit depth of the display (using NVIDIA control panel or the Display control panel on Windows), the amount of memory needed for the primary surface changes. 
> 사용자가 디스플레이의 해상도 또는 비트 심도 (윈도우의 NVIDIA 제어 패널 또는 디스플레이 제어 패널 사용)를 변경하여 디스플레이의 모드 전환을 시작하면 기본 표면에 필요한 메모리 양이 변경됩니다.

For example, if the user changes the display resolution from 1280x1024x32-bit to 1600x1200x32-bit, the system must dedicate 7.68 MB to the primary surface rather than 5.24 MB. (Full-screen graphics applications running with anti-aliasing enabled may require much more display memory for the primary surface.) 
> 예를 들어 사용자가 디스플레이 해상도를 1280x1024x32 비트에서 1600x1200x32 비트로 변경하면 시스템은 5.24MB가 아닌 기본 표면에 7.68MB를 제공해야 합니다. 앤티 앨리어싱을 사용하도록 설정된 전체 화면 그래픽 애플리케이션은 주 표면에 훨씬 많은 디스플레이 메모리가 필요할 수 있습니다.

On Windows, other events that may initiate display mode switches include launching a full-screen DirectX application, hitting Alt+Tab to task switch away from a full-screen DirectX application, or hitting Ctrl+Alt+Del to lock the computer. 
> 윈도우에서 디스플레이 모드 전환을 시작할 수 있는 다른 이벤트로는 전체 화면 DirectX 애플리케이션 론칭, Alt+Tab을 눌러 전체 화면 DirectX 애플리케이션에서 작업 전환 또는 Ctrl+Alt+Del 키를 눌러 컴퓨터 잠그기 등이 있습니다.

If a mode switch increases the amount of memory needed for the primary surface, the system may have to cannibalize memory allocations dedicated to CUDA applications. 
> 모드 스위치가 기본 표면에 필요한 메모리 양을 늘리면 시스템에서 CUDA 애플리케이션 전용 메모리 할당을 동결해야 할 수 있습니다.

Therefore, a mode switch results in any call to the CUDA runtime to fail and return an invalid context error. 
> 따라서 모드 전환으로 인해 CUDA 런타임에 대한 호출이 실패하고 잘못된 컨텍스트 오류가 반환됩니다.

## 3.6 Tesla Compute Cluster Mode for Windows 
> 3.6 윈도우용 Tesla 컴퓨팅 클러스터 모드

Using NVIDIA’s System Management Interface (nvidia-smi), the Windows device driver can be put in TCC (Tesla Compute Cluster) mode for devices of the Tesla and Quadro Series of compute capability 2.0 and higher. 
> NVIDIA의 시스템 관리 인터페이스 (nvidia-smi)를 사용하여 윈도우 디바이스 드라이버를 Tesla 및 Quadro 시리즈의 컴퓨팅 기능 2.0 이상의 디바이스용 TCC (Tesla Compute Cluster) 모드로 전환할 수 있습니다.

This mode has the following primary benefits:  
> 이 모드의 주요 이점은 다음과 같습니다.

It makes it possible to use these GPUs in cluster nodes with non‐NVIDIA integrated graphics;  
> 비 NVIDIA 통합 그래픽이 있는 클러스터 노드에서 이 GPU를 사용할 수 있습니다.

It makes these GPUs available via Remote Desktop, both directly and via cluster management systems that rely on Remote Desktop;  
> 이 GPU는 원격 데스크톱을 통해 직접 또는 원격 데스크톱을 통해 사용하는 클러스터 관리 시스템을 통해 사용할 수 있습니다.

It makes these GPUs available to applications running as a Windows service (i.e. in Session 0). However, the TCC mode removes support for any graphics functionality. 
> 이 GPU를 윈도우 서비스 (즉 세션 0)로 실행되는 애플리케이션에서 사용할 수 있게 합니다. 그러나 TCC 모드는 모든 그래픽 기능에 대한 지원을 제거합니다. 
