## Appendix D. C/C++ Language Support 
> 부록 D. C/C++ 언어 지원

As described in Section 3.1, source files compiled with nvcc can include a mix of host code and device code. 
> 3.1 절에서 설명한 것처럼 nvcc로 컴파일된 소스 파일에는 호스트 코드와 디바이스 코드가 혼합되어 있을 수 있습니다.

For the host code, nvcc supports whatever part of the C++ ISO/IEC 14882:2003 specification the host c++ compiler supports. 
> 호스트 코드의 경우 nvcc는 호스트 c++ 컴파일러가 지원하는 C++ ISO/IEC 14882:2003 명세서의 모든 부분을 지원합니다.

For the device code, nvcc supports the features illustrated in Section D.1 with some restrictions described in Section D.2; 
> 디바이스 코드의 경우 nvcc는 D.2 절에서 설명한 몇 가지 제한사항을 사용하여 D.1 절에 나와있는 기능을 지원합니다.

it does not support run time type information (RTTI), exception handling, and the C++ Standard Library. 
> 런타임 유형 정보 (RTTI), 예외 처리 및 C++ 표준 라이브러리를 지원하지 않습니다.

## D.1 Code Samples 
> D.1 코드 샘플

## D.1.1 Data Aggregation Class 
> D.1.1 데이터 집계 클래스

## D.1.2 Derived Class 
> D.1.2 파생 클래스

## D.1.4 Function Template 
> D.1.4 함수 템플릿

## D.1.5 Functor Class 
> D.1.5 펑터 클래스

## D.2 Restrictions 
> D.2 제한

## D.2.1 Qualifiers 
> D.2.1 한정자 

## D.2.1.1 Device Memory Qualifiers
> D.2.1.1 디바이스 메모리 한정자

The __device__, __shared__ and __constant__ qualifiers are not allowed on:  
> __device__, __shared__ 및 __constant__ 한정자는 다음에서 사용할 수 없습니다.

class, struct, and union data members,  formal parameters,  local variables within a function that executes on the host. 
> 클래스, 구조체 및 공용체 데이터 멤버, 형식 매개변수, 호스트에서 실행되는 함수 내의 로컬 변수.

__shared__ and __constant__ variables have implied static storage. 
> __shared__ 및 __constant__ 변수는 정적 저장을 의미합니다.

__device__ and __constant__ variables are only allowed at file scope. 
> __device__ 및 __constant__ 변수는 파일 범위에서만 허용됩니다.

__device__, __shared__ and __constant__ variables cannot be defined as external using the extern keyword. 
> __device__, __shared__ 및 __constant__ 변수는 extern 키워드를 사용하여 외부 변수로 정의할 수 없습니다.

The only exception is for dynamically allocated __shared__ variables as described in Section B.2.3. 
> 유일한 예외는 B.2.3 절에서 설명한 것처럼 동적으로 할당된 __shared__ 변수입니다.

## D.2.1.2 Volatile Qualifier 
> D.2.1.2 불안정한 한정자

Only after the execution of a __threadfence_block(), __threadfence(), or __syncthreads() (Sections B.5 and B.6) are prior writes to global or shared memory guaranteed to be visible by other threads. 
> __threadfence_block(), __threadfence() 또는 __syncthreads() (B.5 및 B.6 절)는 전역 또는 공유 메모리에 대한 사전 쓰기므로 다른 스레드에서 볼 수 있습니다.

As long as this requirement is met, the compiler is free to optimize reads and writes to global or shared memory. This behavior can be changed using the volatile keyword: 
> 이 요구 사항이 충족되는 한 컴파일러는 전역 또는 공유 메모리에 대한 읽기 및 쓰기를 자유롭게 최적화할 수 있습니다. 이 동작은 volatile 키워드를 사용하여 변경할 수 있습니다.

If a variable located in global or shared memory is declared as volatile, the compiler assumes that its value can be changed or used at any time by another thread and therefore any reference to this variable compiles to an actual memory read or write instruction. 
> 전역 메모리 또는 공유 메모리에 있는 변수가 volatile로 선언되면 컴파일러는 다른 스레드가 언제든지 값을 변경하거나 사용할 수 있다고 가정하므로 이 변수에 대한 참조는 실제 메모리 읽기 또는 쓰기 명령어로 컴파일됩니다.

For example, in the code sample of Section 5.4.3, if s_ptr were not declared as volatile, the compiler would optimize away the store to shared memory for each assignment to s_ptr[tid]. 
> 예를 들어, 5.4.3 절의 코드 샘플에서 s_ptr이 volatile로 선언되지 않으면 컴파일러는 s_ptr[tid]에 대한 각 할당에 대해 저장소를 공유 메모리로 최적화합니다.

It would accumulate the result into a register instead and only store the final result to shared memory, which would be incorrect. 
> 대신 결과를 레지스터에 누적하여 최종 결과를 공유 메모리에만 저장합니다. 이는 잘못된 것입니다.

## D.2.2 Pointers 
> D.2.2 포인터

For devices of compute capability 1.x, pointers in code that is executed on the device are supported as long as the compiler is able to resolve whether they point to either the shared memory space, the global memory space, or the local memory space, otherwise they are restricted to only point to memory allocated or declared in the global memory space. 
> 컴퓨팅 기능 1.x의 디바이스인 경우 디바이스에서 실행되는 코드의 포인터는 컴파일러가 공유 메모리 공간, 전역 메모리 공간 또는 로컬 메모리 공간을 가리키는지 여부를 확인할 수 있는 한 지원됩니다. 그렇지 않으면 전역 메모리 공간에 할당되거나 선언된 메모리를 가리키도록 제한됩니다.

For devices of compute capability 2.x and higher, pointers are supported without any restriction. 
> 컴퓨팅 성능이 2.x 이상인 디바이스의 경우 포인터가 아무런 제한없이 지원됩니다.

Dereferencing a pointer either to global or shared memory in code that is executed on the host, or to host memory in code that is executed on the device results in an undefined behavior, most often in a segmentation fault and application termination. 
> 호스트에서 실행되는 코드에서 전역 또는 공유 메모리로 포인터를 역참조하거나 디바이스에서 실행되는 코드의 호스트 메모리를 지정하면 정의되지 않은 동작이 발생합니다. 대부분 세그먼트 오류 및 애플리케이션이 발생합니다.

The address obtained by taking the address of a __device__, __shared__ or __constant__ variable can only be used in device code. 
> __device__, __shared__ 또는 __constant__ 변수의 주소를 사용하여 얻은 주소는 디바이스 코드에서만 사용할 수 있습니다.

## D.2.3 Operators 
> D.2.3 연산자

## D.2.3.1 Assignment Operator 
> D.2.3.1 할당 연산자

__constant__ variables can only be assigned from the host code through runtime functions (Sections 3.2.2); they cannot be assigned from the device code. __shared__ variables cannot have an initialization as part of their declaration. 
> __constant__ 변수는 런타임 기능을 통해 호스트 코드에서만 할당할 수 있습니다 (3.2.2 절). 디바이스 코드에서 할당할 수 없습니다. __shared__ 변수는 선언의 일부로 초기화를 가질 수 없습니다.

It is not allowed to assign values to any of the built-in variables defined in Section B.4. 
> B.4 절에 정의된 내장 변수에 값을 할당할 수 없습니다.

The address of a __device__ or __constant__ variable obtained through cudaGetSymbolAddress() as described in Section 3.2.2 can only be used in host code. 
> 섹션 3.2.2에서 설명한 대로 cudaGetSymbolAddress()를 통해 얻은 __device__ 또는 __constant__ 변수의 주소는 호스트 코드에서만 사용할 수 있습니다.

## D.2.3.2 Address Operator 
> D.2.3.2 주소 연산자

It is not allowed to take the address of any of the built-in variables defined in Section B.4. 
> B.4 절에 정의된 내장 변수 중 하나의 주소를 취하는 것은 허용되지 않습니다.

## D.2.4 Functions 
> D.2.4 함수

## D.2.4.1 Function Parameters 
> D.2.4.1 함수 매개변수

__global__ function parameters are passed to the device:  via shared memory and are limited to 256 bytes on devices of compute capability 1.x,  via constant memory and are limited to 4 KB on devices of compute capability 2.x and higher. 
> __global__ 함수 매개변수가 디바이스에 전달됩니다. 공유 메모리를 통해 전송되며 컴퓨팅 기능 1.x의 디바이스에서는 256 바이트로 제한되고, 상수 메모리를 통해 전송되며 컴퓨팅 기능이 2.x 이상인 디바이스에서는 4KB로 제한됩니다.

__device__ and __global__ functions cannot have a variable number of arguments. __device__ 및 __global__ 함수는 가변 인자를 가질 수 없습니다.
 
## D.2.4.2 Static Variables within Function 
> D.2.4.2 함수 내의 정적 변수 

Static variables cannot be declared within the body of __device__ and __global__ functions. 
> 정적 변수는 __device__ 및 __global__ 함수의 본문 내에서 선언할 수 없습니다.

## D.2.4.3 Function Pointers 
> D.2.4.3 함수 포인터

Function pointers to __global__ functions are supported in host code, but not in device code. 
> __global__ 함수에 대한 함수 포인터는 호스트 코드에서 지원되지만 디바이스 코드에서는 지원되지 않습니다.

Function pointers to __device__ functions are only supported in device code compiled for devices of compute capability 2.x and higher. It is not allowed to take the address of a __device__ function in host code. 
> __device__ 함수에 대한 함수 포인터는 컴퓨팅 기능 2.x 이상의 디바이스용으로 컴파일된 디바이스 코드에서만 지원됩니다. 호스트 코드에서 __device__ 함수의 주소를 가져올 수 없습니다.

## D.2.4.4 Function Recursion 
> D.2.4.4 함수 재귀

__global__ functions do not support recursion. 
> __global__ 함수는 재귀를 지원하지 않습니다. 

__device__ functions only support recursion in device code compiled for devices of compute capability 2.x and higher. 
> __device__ 함수는 컴퓨팅 기능이 2.x 이상의 디바이스인 컴파일된 디바이스 코드에서만 재귀를 지원합니다.

## D.2.5 Classes 
> D.2.5 클래스

## D.2.5.1 Data Members 
> D.2.5.1 데이터 멤버

Static data members are not supported. 
> 정적 데이터 멤버는 지원되지 않습니다.

The layout of bit-fields in device code may currently not match the layout in host code on Windows. 
> 디바이스 코드의 비트 필드 레이아웃은 현재 윈도우의 호스트 코드 레이아웃과 일치하지 않을 수 있습니다.

## D.2.5.2 Function Members 
> D.2.5.2 함수 멤버

Static member functions cannot be __global__ functions. 
> 정적 멤버 함수는 __global__ 함수가 될 수 없습니다.

## D.2.5.3 Constructors and Destructors 
> D.2.5.3 생성자와 소멸자 

Declaring global variables for which a constructor or a destructor needs to be called is not supported.  
> 생성자 또는 소멸자를 호출해야 하는 전역 변수를 선언하는 것은 지원되지 않습니다.

## D.2.5.4 Virtual Functions 
> D.2.5.4 가상 함수

Declaring global variables of a class with virtual functions is not supported.  
> 가상 함수를 가진 클래스의 전역 변수를 선언하는 것은 지원되지 않습니다.

It is not allowed to pass as an argument to a __global__ function an object of a class with virtual functions. 
> 가상 함수가 있는 클래스의 객체를 __global__ 함수에 변수로 전달할 수 없습니다.

The virtual function table is placed in global or constant memory by the compiler. 
> 가상 함수 표는 컴파일러에 의해 전역 또는 상수 메모리에 배치됩니다.

## D.2.5.5 Virtual Base Classes 
> D.2.5.5 가상 기본 클래스

It is not allowed to pass as an argument to a __global__ function an object of a class derived from virtual base classes. 
> 가상 기본 클래스에서 파생된 클래스의 객체를 __global__ 함수에 변수로 전달할 수 없습니다.

## D.2.5.6 Windows-Specific 
> D.2.5.6 윈도우 전용

On Windows, the CUDA compiler may produce a different memory layout, compared to the host Microsoft compiler, for a C++ object of class type T that satisfies any of the following conditions:  
> 윈도우에서 다음 조건 중 하나를 충족하는 클래스 유형 T의 C++ 객체에 대한 호스트 마이크로소프트 컴파일러에 비해, CUDA 컴파일러는 다른 메모리 레이아웃을 생성할 수 있습니다.

T has virtual functions or derives from a direct or indirect base class that has virtual functions; 
> T는 가상 함수가 있거나 가상 함수가 있는 직접 또는 간접 기본 클래스에서 파생됩니다. 
 
T has a direct or indirect virtual base class;  
> T는 직접 또는 간접 가상 기본 클래스를 가집니다.

T has multiple inheritance with more than one direct or indirect empty base class. 
> T는 하나 이상의 직접 또는 간접 빈 기본 클래스가 있는 다중 상속(계승)을 가집니다.

The size for such an object may also be different in host and device code. 
> 이러한 객체의 크기는 호스트 및 디바이스 코드에 따라 다를 수 있습니다.

As long as type T is used exclusively in host or device code, the program should work correctly. 
> 유형 T가 호스트 또는 디바이스 코드에서 독점적으로 사용되는 한 프로그램은 올바르게 작동해야 합니다.

Do not pass objects of type T between host and device code (e.g. as arguments to __global__ functions or through cudaMemcpy*() calls). 
> 호스트와 디바이스 코드 사이에 유형 T의 객체를 전달하지 마십시오 (예를 들면, __global__ 함수의 변수 또는 cudaMemcpy*() 호출을 통해).

## D.2.6 Templates
> D.2.6 템플릿
 
A __global__ function template cannot be instantiated with a type or typedef that is defined within a function or is private to a class or structure, as illustrated in the following code sample: 
> __global__ 함수 템플릿은 함수 내에 정의된 유형이나 typedef로 인스턴스화할 수 없으며 다음 코드 샘플에서 설명하는 것처럼 클래스나 구조체에 대해 비공개입니다.
