## Appendix G. Driver API 
> 부록 G. 드라이버 API

This appendix assumes knowledge of the concepts described in Section 3.2. 
> 이 부록은 3.2 절에서 설명된 개념을 알고 있다고 가정합니다.

The driver API is implemented in the nvcuda dynamic library which is copied on the system during the installation of the device driver. All its entry points are prefixed with cu. 
> 드라이버 API는 디바이스 드라이버를 설치하는 동안 시스템에 복사되는 nvcuda 동적 라이브러리에 구현됩니다. 모든 진입점 앞에는 cu가 붙습니다.

It is a handle-based, imperative API: Most objects are referenced by opaque handles that may be specified to functions to manipulate the objects. 
> 이것은 핸들 기반의 명령형 API입니다. 대부분의 객체는 객체를 조작하는 함수에 지정될 수 있는 불투명 핸들에 의해 참조됩니다.

The objects available in the driver API are summarized in Table G-1. 
> 드라이버 API에서 사용할 수있는 객체는 표 G-1에 요약되어 있습니다.

Table G-1. Objects Available in the CUDA Driver API 
> 표 G-1. CUDA 드라이버 API에서 사용할 수있는 객체

The driver API must be initialized with cuInit() before any function from the driver API is called. 
> 드라이버 API의 함수가 호출되기 전에 드라이버 API를 cuInit()로 초기화해야 합니다.

A CUDA context must then be created that is attached to a specific device and made current to the calling host thread as detailed in Section G.1. 
> 그런 다음 G.1 절에서 설명하는 대로 특정 디바이스에 연결되고 호출하는 호스트 스레드에 대해 최신 상태로 만든 CUDA 컨텍스트를 작성해야 합니다.

Within a CUDA context, kernels are explicitly loaded as PTX or binary objects by the host code as described in Section G.2. 
> CUDA 컨텍스트 내에서 커널은 G.2 절에서 설명한 대로 호스트 코드에 의해 PTX 또는 바이너리 객체로 명시적으로 로드됩니다.

Kernels written in C must therefore be compiled separately into PTX or binary objects. 
> 따라서 C로 작성된 커널은 PTX 또는 2 진 객체로 별도로 컴파일해야 합니다.

Kernels are launched using API entry points as described in Section G.3. 
> 커널은 G.3 절에서 설명한 대로 API 진입점을 사용하여 시작됩니다.

Any application that wants to run on future device architectures must load PTX, not binary code. 
> 향후 디바이스 아키텍처에서 실행하고자 하는 모든 애플리케이션은 바이너리 코드가 아닌 PTX를 로드해야 합니다.

This is because binary code is architecture-specific and therefore incompatible with future architectures, whereas PTX code is compiled to binary code at load time by the device driver. 
> 이는 바이너리 코드가 아키텍처에 따라 다르므로 향후 아키텍처와 호환되지 않기 때문에 PTX 코드는 디바이스 드라이버에 의해 로드시 바이너리 코드로 컴파일되기 때문입니다.

Here is the host code of the sample from Section 2.1 written using the driver API: 
> 다음은 드라이버 API를 사용하여 작성한 2.1 절의 샘플 호스트 코드입니다.

Full code can be found in the vectorAddDrv SDK code sample. 
> 전체 코드는 vectorAddDrv SDK 코드 샘플에서 찾을 수 있습니다.

## G.1 Context 
> G.1 컨텍스트

A CUDA context is analogous to a CPU process. 
> CUDA 컨텍스트는 CPU 프로세스와 유사합니다.

All resources and actions performed within the driver API are encapsulated inside a CUDA context, and the system automatically cleans up these resources when the context is destroyed. 
> 드라이버 API 내에서 수행되는 모든 리소스와 동작은 CUDA 컨텍스트에 캡슐화되며 컨텍스트가 파괴되면 시스템은 자동으로 이러한 리소스를 정리합니다.

Besides objects such as modules and texture or surface references, each context has its own distinct address space. 
> 모듈 및 텍스처 또는 표면 참조와 같은 객체 외에도 각 컨텍스트에는 고유한 자체 주소 공간이 있습니다.

As a result, CUdeviceptr values from different contexts reference different memory locations. 
> 결과적으로 다른 컨텍스트의 CUdeviceptr 값은 서로 다른 메모리 위치를 참조합니다.

A host thread may have only one device context current at a time. 
> 호스트 스레드는 한 번에 하나의 디바이스 컨텍스트 전류만 가질 수 있습니다.

When a context is created with cuCtxCreate(), it is made current to the calling host thread. 
> cuCtxCreate()를 사용하여 컨텍스트를 만들면 호출하는 호스트 스레드에 컨텍스트가 만들어집니다.

CUDA functions that operate in a context (most functions that do not involve device enumeration or context management) will return CUDA_ERROR_INVALID_CONTEXT if a valid context is not current to the thread. 
> 컨텍스트에서 작동하는 CUDA 함수 (디바이스 열거 또는 컨텍스트 관리를 포함하지 않는 대부분의 함수)는 유효한 컨텍스트가 스레드에 최신이 아닌 경우 CUDA_ERROR_INVALID_CONTEXT를 반환합니다.

Each host thread has a stack of current contexts. cuCtxCreate() pushes the new context onto the top of the stack. cuCtxPopCurrent() may be called to detach the context from the host thread. 
> 각 호스트 스레드에는 현재 컨텍스트 스택이 있습니다. cuCtxCreate()는 새 컨텍스트를 스택 맨 위로 푸시합니다. 호스트 스레드에서 컨텍스트를 분리하기 위해 cuCtxPopCurrent()를 호출할 수 있습니다.

The context is then "floating" and may be pushed as the current context for any host thread. cuCtxPopCurrent() also restores the previous current context, if any. 
> 그러면 컨텍스트는 "유동"하며 모든 호스트 스레드에 대한 현재 컨텍스트로 푸시될 수 있습니다. cuCtxPopCurrent()는 이전 컨텍스트 (있는 경우)도 복원합니다.

A usage count is also maintained for each context. cuCtxCreate() creates a context with a usage count of 1. 
> 사용 카운트(횟수)도 각 컨텍스트를 위해 유지됩니다. cuCtxCreate()는 사용 횟수가 1 인 컨텍스트를 만듭니다.

cuCtxAttach() increments the usage count and  cuCtxDetach() decrements it. 
> cuCtxAttach()는 사용 횟수를 증가시키고 cuCtxDetach()는이를 감소시킵니다.

A context is destroyed when the usage count goes to 0 when calling cuCtxDetach() or cuCtxDestroy(). 
> 컨텍스트는 cuCtxDetach() 또는 cuCtxDestroy()를 호출할 때 사용 카운트가 0이 되면 파괴됩니다.

Usage count facilitates interoperability between third party authored code operating in the same context. 
> 사용 카운트(횟수)는 동일한 컨텍스트에서 작동하는 타사(제3자) 제작 코드 간의 상호 운용성을 용이하게 합니다.

For example, if three libraries are loaded to use the same context, each library would call cuCtxAttach() to increment the usage count and cuCtxDetach() to decrement the usage count when the library is done using the context. 
> 예를 들어 동일한 컨텍스트를 사용하기 위해 세 개의 라이브러리가 로드된 경우 각 라이브러리는 cuCtxAttach()를 호출하여 사용 횟수를 증가시키고 cuCtxDetach()는 라이브러리가 컨텍스트를 사용하여 완료될 때 사용 횟수를 감소시킵니다.

For most libraries, it is expected that the application will have created a context before loading or initializing the library; that way, the application can create the context using its own heuristics, and the library simply operates on the context handed to it. 
> 대부분의 라이브러리에서는 애플리케이션이 라이브러리를 로드하거나 초기화하기 전에 컨텍스트를 만들어야 합니다. 그런 식으로 애플리케이션은 자체 발견적 방법을 사용하여 컨텍스트를 만들 수 있으며 라이브러리는 손쉽게 컨텍스트에서 작동합니다.

Libraries that wish to create their own contexts – unbeknownst to their API clients who may or may not have created contexts of their own – would use cuCtxPushCurrent() and cuCtxPopCurrent() as illustrated in Figure G-1. 
> 자체 컨텍스트를 생성하고자 하는 라이브러리 - 자체 컨텍스트를 생성했거나 생성하지 않았던 API 클라이언트와 모르는 경우 - 그림 G-1에서 설명한 것처럼 cuCtxPushCurrent() 및 cuCtxPopCurrent()를 사용합니다.
 
Figure G-1  Library Context Management 
> 그림 G-1 라이브러리 컨텍스트 관리

## G.2 Module 
> G.2 모듈

Modules are dynamically loadable packages of device code and data, akin to DLLs in Windows, that are output by nvcc (see Section 3.1). 
> 모듈은 nvcc (3.1 절 참조)가 출력하는 윈도우의 DLL과 비슷한 동적으로 로드 가능한 디바이스 코드 및 데이터 패키지입니다.

The names for all symbols, including functions, global variables, and texture or surface references, are maintained at module scope so that modules written by independent third parties may interoperate in the same CUDA context. 
> 함수, 전역 변수, 텍스처 또는 표면 참조를 포함한 모든 심볼의 이름은 모듈 범위에서 유지 관리되므로 독립 제 3자가 작성한 모듈이 동일한 CUDA 컨텍스트에서 상호운용될 수 있습니다.

This code sample loads a module and retrieves a handle to some kernel: 
> 이 코드 샘플은 모듈을 로드하고 일부 커널에 대한 처리를 검색합니다.

This code sample compiles and loads a new module from PTX code and parses compilation errors: 
> 이 코드 샘플은 PTX 코드에서 새 모듈을 컴파일하고 로드하고 컴파일 오류를 파싱(구문 분석) 합니다.

## G.3 Kernel Execution 
> G.3 커널 실행

cuLaunchKernel() launches a kernel with a given execution configuration. 
> cuLaunchKernel()은 지정된 실행 설정으로 커널을 시작합니다.
 
Parameters are passed either as an array of pointers (next to last parameter of cuLaunchKernel()) where the nth pointer corresponds to the nth parameter and points to a region of memory from which the parameter is copied, or as one of the extra options (last parameter of cuLaunchKernel()).  
> 매개변수는 n 번째 포인터는 n 번째 매개변수에 해당하며 매개변수가 복사되는 메모리 영역을 가리키거나 추가 옵션 중 하나 (cuLaunchKernel()의 마지막 매개변수)를 가리키는 포인터 배열 (cuLaunchKernel()의 마지막 매개변수 옆)으로 전달됩니다.

When parameters are passed as an extra option (the CU_LAUNCH_PARAM_BUFFER_POINTER option), they are passed as a pointer to a single buffer where parameters are assumed to be properly offset with respect to each other by matching the alignment requirement for each parameter type in device code. 
> 매개변수가 추가 옵션 (CU_LAUNCH_PARAM_BUFFER_POINTER 옵션)으로 전달되면 매개변수는 디바이스 코드의 각 매개변수 유형에 대한 정렬 요구 사항을 일치시켜 매개변수가 서로 적절하게 오프셋된 단일 버퍼에 대한 포인터로 전달됩니다.

Alignment requirements in device code for the built-in vector types are listed in Table B-1. 
> 내장된 벡터 유형에 대한 디바이스 코드의 정렬 요구 사항은 표 B-1에 나열되어 있습니다.

For all other basic types, the alignment requirement in device code matches the alignment requirement in host code and can therefore be obtained using __alignof(). 
> 다른 모든 기본 유형의 경우, 디바이스 코드의 정렬 요구 사항은 호스트 코드의 정렬 요구 사항과 일치하므로 __alignof()를 사용하여 얻을 수 있습니다.

The only exception is when the host compiler aligns double and long long (and long on a 64-bit system) on a one-word boundary instead of a two-word boundary (for example, using gcc’s compilation flag -mno-aligndouble) since in device code these types are always aligned on a two-word boundary. 
> 유일한 예외는 디바이스 코드에서 이러한 유형은 항상 2 단어 경계에 정렬되기 때문에 호스트 컴파일러가 2 단어 경계 대신 1 단어 경계 (예를 들어 gcc의 컴파일 플래그 -mno-aligndouble를 사용)에서 double long long (64 비트 시스템에서는 long)을 정렬하는 경우입니다. 

CUdeviceptr is an integer, but represents a pointer, so its alignment requirement is __alignof(void*). 
> CUdeviceptr은 정수이지만 포인터를 나타내기 때문에 정렬 요구 사항은 __alignof(void*)입니다.

The following code sample uses a macro (ALIGN_UP()) to adjust the offset of each parameter to meet its alignment requirement and another macro (ADD_TO_PARAM_BUFFER()) to add each parameter to the parameter buffer passed to the CU_LAUNCH_PARAM_BUFFER_POINTER option. 
> 다음 코드 샘플에서는 매크로 (ALIGN_UP())를 사용하여 정렬 요구 사항을 충족시키는 각 매개변수의 오프셋을 조정하고 다른 매크로 (ADD_TO_PARAM_BUFFER())를 사용하여 각 매개변수를 CU_LAUNCH_PARAM_BUFFER_POINTER 옵션에 전달된 매개변수 버퍼에 추가합니다.

The alignment requirement of a structure is equal to the maximum of the alignment requirements of its fields. 
> 구조체의 정렬 요구 사항은 해당 필드의 정렬 요구 사항의 최대값과 같습니다.

The alignment requirement of a structure that contains built-in vector types, CUdeviceptr, or non-aligned double and long long, might therefore differ between device code and host code. 
> 따라서 기본 제공 벡터 유형인 CUdeviceptr 또는 정렬되지 않은 double 및 long long을 포함하는 구조의 정렬 요구 사항은 디바이스 코드와 호스트 코드 간에 다를 수 있습니다.

Such a structure might also be padded differently. 
> 이러한 구조는 다르게 패딩될 수도 있습니다.

The following structure, for example, is not padded at all in host code, but it is padded in device code with 12 bytes after field f since the alignment requirement for field f4 is 16.  
> 예를 들어 다음 구조는 호스트 코드에서 전혀 패딩되지 않지만 필드 f4의 정렬 요구 사항이 16이므로 디바이스 코드에서 필드 f 다음 12 바이트로 채워집니다.

## G.4 Interoperability between Runtime and Driver APIs 
> G.4 런타임 API와 드라이버 API 간의 상호 운용성

An application can mix runtime API code with driver API code. 
> 애플리케이션은 런타임 API 코드와 드라이버 API 코드를 혼합할 수 있습니다.

If a context is created and made current via the driver API, subsequent runtime calls will pick up this context instead of creating a new one. 
> 컨텍스트가 생성되어 드라이버 API를 통해 최신 상태가 되면 후속 런타임 호출은 새로운 컨텍스트를 생성하는 대신 이 컨텍스트를 선택합니다.

If the runtime is initialized (implicitly as mentioned in Section 3.2), cuCtxGetCurrent() can be used to retrieve the context created during initialization. 
> 런타임이 초기화되면 (암시적으로 3.2 절에서 언급됨), cuCtxGetCurrent()를 사용하여 초기화하는 동안 생성된 컨텍스트를 검색할 수 있습니다.

This context can be used by subsequent driver API calls. 
> 이 컨텍스트는 후속 드라이버 API 호출에서 사용할 수 있습니다.

Device memory can be allocated and freed using either API. CUdeviceptr can be cast to regular pointers and vice-versa: 
> 디바이스 메모리는 API 중 하나를 사용하여 할당 및 해제할 수 있습니다. CUdeviceptr은 일반 포인터로 캐스트될 수 있으며 그 반대도 가능합니다.

In particular, this means that applications written using the driver API can invoke libraries written using the runtime API (such as CUFFT, CUBLAS, …). 
> 특히 이는 드라이버 API를 사용하여 작성된 애플리케이션이 런타임 API (CUFFT, CUBLAS ... 같은)를 사용하여 작성된 라이브러리를 호출할 수 있음을 의미합니다.

All functions from the device and version management sections of the reference manual can be used interchangeably. 
> 참조 메뉴얼의 디바이스 및 버전 관리 섹션의 모든 기능은 교환 가능하게 사용할 수 있습니다.
 
