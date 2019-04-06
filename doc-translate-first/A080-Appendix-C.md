# Appendix C. Mathematical Functions 
> 부록 C. 수학 함수

The reference manual lists, along with their description, all the functions of the C/C++ standard library mathematical functions that are supported in device code, as well as all intrinsic functions (that are only supported in device code). 
> 참조 설명서에는 설명과 함께 디바이스 코드에서 지원되는 C/C++ 표준 라이브러리 수학 함수의 모든 기능과 디바이스 코드에서만 지원되는 모든 내장 함수가 나열되어 있습니다.

This appendix provides accuracy information for some of these functions when applicable. 
> 이 부록에서는 적용 가능한 경우 이러한 기능 중 일부에 대한 정확성 정보를 제공합니다.

## C.1 Standard Functions
> C.1 표준 함수
 
The functions from this section can be used in both host and device code. 
> 이 섹션의 기능은 호스트 및 디바이스 코드에서 모두 사용할 수 있습니다.

This section specifies the error bounds of each function when executed on the device and also when executed on the host in the case where the host does not supply the function. 
> 이 섹션은 디바이스에서 실행될 때 각 기능의 오류 범위를 지정하며 호스트가 기능을 제공하지 않는 경우 호스트에서 실행될 때도 지정합니다.

The error bounds are generated from extensive but not exhaustive tests, so they are not guaranteed bounds. 
> 오류 경계는 광범위하게 생성되지만 철저한 테스트는 아니므로 보증 범위가 아닙니다.

## C.1.1 Single-Precision Floating-Point Functions 
> C.1.1 단 정밀도 부동 소수점 함수

Addition and multiplication are IEEE-compliant, so have a maximum error of 0.5 ulp. 
> 더하기 및 곱셈은 IEEE 규격이므로 최대 오차는 0.5 ulp입니다.

However, on the device, the compiler often combines them into a single multiply-add instruction (FMAD) and for devices of compute capability 1.x, FMAD truncates the intermediate result of the multiplication as mentioned in Section F.2. 
> 그러나 디바이스에서 컴파일러는 종종 이를 FMAD (multiply-add instruction)로 결합하고 컴퓨팅 기능 1.x인 디바이스의 경우 FMAD는 F.2 절에서 설명한 대로 곱셈의 중간 결과를 잘라냅니다.

This combination can be avoided by using the __fadd_[rn,rz,ru,rd]() and __fmul_[rn,rz,ru,rd]() intrinsic functions (see Section C.2). 
> 이 조합은 __fadd_ [rn,rz,ru,rd]() 및 __fmul_ [rn,rz,ru,rd]() 내장 함수를 사용하여 피할 수 있습니다 (C.2 절 참조).

The recommended way to round a single-precision floating-point operand to an integer, with the result being a single-precision floating-point number is rintf(), not roundf(). 
> 단 정밀도 부동 소수점 피연산자를 정수로 반올림하고 결과가 단 정밀도 부동 소수점 숫자가 되도록하는 추천 방법은 roundf()가 아니라 rintf()입니다.

The reason is that roundf() maps to an 8-instruction sequence on the device, whereas rintf() maps to a single instruction. truncf(), ceilf(), and floorf() each map to a single instruction as well. 
> 이유는 roundf()가 디바이스의 8 개 명령 시퀀스에 매핑되는 반면 rintf()는 단일 명령어에 매핑되기 때문입니다. truncf(), ceilf() 및 floorf()는 각각 하나의 명령에도 매핑됩니다.

Table C-1. Mathematical Standard Library Functions with Maximum ULP Error 
> 표 C-1. 최대 ULP 오류가 있는 수학 표준 라이브러리 함수

The maximum error is stated as the absolute value of the difference in ulps between a correctly rounded single-precision result and the result returned by the CUDA library function. 
> 최대 오류는 정확하게 반올림한 단 정밀도 결과와 CUDA 라이브러리 함수에 의해 반환된 결과 사이의 ulps 차이의 절대값으로 표시됩니다.

## C.1.2 Double-Precision Floating-Point Functions 
> C.1.2 배 정밀도 부동 소수점 함수

The errors listed below only apply when compiling for devices with native double-precision support. 
> 아래에 나열된 오류는 기본 배 정밀도 지원이 있는 디바이스용으로 컴파일할 때만 적용됩니다.

When compiling for devices without such support, such as devices of compute capability 1.2 and lower, the double type gets demoted to float by default and the double-precision math functions are mapped to their single-precision equivalents. 
> 이러한 지원이 없는 디바이스 (컴퓨팅 기능이 1.2 이하인 디바이스 같은)를 컴파일할 때 double 유형은 기본적으로 부동화되도록 강등되고 배 정밀도 math 함수는 단 정밀도 해당 함수에 매핑됩니다.

The recommended way to round a double-precision floating-point operand to an integer, with the result being a double-precision floating-point number is rint(), not round(). 
> 배 정밀도 부동 소수점 피연산자를 정수로 반올림하고 결과가 배 정밀도 부동 소수점 숫자가 되도록 하는 추천 방법은 round()가 아니라 rint()입니다.

The reason is that round() maps to an 8-instruction sequence on the device, whereas rint() maps to a single instruction. trunc(), ceil(), and floor() each map to a single instruction as well. 
> 그 이유는 round()가 디바이스의 8 개 명령 시퀀스에 매핑되는 반면 rint()는 단일 명령어에 매핑되기 때문입니다. trunc(), ceil() 및 floor()는 각각 하나의 명령에도 매핑됩니다.

Table C-2.  Mathematical Standard Library Functions with Maximum ULP Error 
> 표 C-2. 최대 ULP 오류가 있는 수학 표준 라이브러리 함수

The maximum error is stated as the absolute value of the difference in ulps between a correctly rounded double-precision result and the result returned by the CUDA library function. 
> 최대 오류는 올바르게 반올림된 배 정밀도 결과와 CUDA 라이브러리 함수에서 반환한 결과 간의 ulps 차이 절대값으로 표시됩니다.
 
The functions from this section can only be used in device code. 
> 이 섹션의 기능은 디바이스 코드에서만 사용할 수 있습니다.

Among these functions are the less accurate, but faster versions of some of the functions of Section C.1. 
> 이 기능들 중에는 C.1 절의 일부 기능의 정확성은 떨어지지만 빠른 버전이 있습니다.

They have the same name prefixed with __ (such as __sinf(x)). 
> 동일한 이름 앞에 __가 붙습니다 (__sinf(x) 같은).

They are faster as they map to fewer native instructions. 
> 적은 기본 명령어로 매핑되므로 속도가 빠릅니다.

The compiler has an option (-use_fast_math) that forces each function in Table C-3 to compile to its intrinsic counterpart. 
> 컴파일러에는 표 C-3의 각 함수가 고유한 부분으로 컴파일되도록 하는 옵션 (-use_fast_math)이 있습니다.

In addition to reducing the accuracy of the affected functions, it may also cause some differences in special case handling. 
> 영향을 받는 기능의 정확도를 줄이는 것 외에도 특수 케이스 처리에 약간의 차이가 발생할 수 있습니다.

A more robust approach is to selectively replace mathematical function calls by calls to intrinsic functions only where it is merited by the performance gains and where changed properties such as reduced accuracy and different special case handling can be tolerated. 
> 좀 더 강력한 접근법은 정밀도가 낮아지고 특수한 케이스 취급이 다를 수도 있는 것 같은 성능 향상으로 이점을 얻을 수 있고 변경된 속성을 허용할 수 있는 곳에서 intrinsic 함수로만 호출하여 수학 함수 호출을 선택적으로 대체하는 것입니다. 

Functions suffixed with _rn operate using the round-to-nearest-even rounding mode. 
> _rn 접미사가 붙은 함수는 가장 가까운 수로 반올림 모드를 사용하여 작동합니다. 

Functions suffixed with _rz operate using the round-towards-zero rounding mode. 
> _rz 접미사가 붙은 함수는 0으로 반올림 모드를 사용하여 작동합니다.

Functions suffixed with _ru operate using the round-up (to positive infinity) rounding mode. 
> 접미사가 _ru 인 함수는 반올림 모드 (양의 무한대로) 올림 모드를 사용하여 작동합니다.

Functions suffixed with _rd operate using the round-down (to negative infinity) rounding mode. 
> 접미사 _rd로 끝나는 함수는 반올림 모드 (음수 무한대로) 반올림 모드를 사용하여 작동합니다.

## C.2.1 Single-Precision Floating-Point Functions 
> C.2.1 단 정밀도 부동 소수점 함수

__fadd_[rn,rz,ru,rd]() and __fmul_[rn,rz,ru,rd]() map to addition and multiplication operations that the compiler never merges into FMADs. 
> __fadd_[rn,rz,ru,rd]() 및 __fmul_[rn,rz,ru,rd]()는 컴파일러가 FMAD에 병합하지 않는 덧셈 및 곱셈 연산에 매핑됩니다.

By contrast, additions and multiplications generated from the '*' and '+' operators will frequently be combined into FMADs. 
> 대조적으로 '*'및 '+'연산자에서 생성된 덧셈 및 곱셈은 자주 FMAD에 결합됩니다.

The accuracy of floating-point division varies depending on the compute capability of the device and whether the code is compiled with -prec-div=false or -prec-div=true. 
> 부동 소수점 나누기의 정확도는 디바이스의 컴퓨팅 기능 및 코드가 -prec-div=false 또는 -prec-div=true로 컴파일되는지 여부에 따라 다릅니다.

For devices of compute capability 1.x or for devices of compute capability 2.x and higher when the code is compiled with -prec-div=false, both the regular division “/” operator and __fdividef(x,y) have the same accuracy, but for 2126 < y < 2128, __fdividef(x,y) delivers a result of zero, whereas the “/” operator delivers the correct result to within the accuracy stated in Table C-4. 
> 코드가 -prec-div=false로 컴파일될 때 컴퓨팅 기능 1.x의 디바이스 또는 컴퓨팅 기능 2.x 이상인 디바이스의 경우 일반 분할 "/" 연산자와 __fdividef(x, y)가 모두 동일합니다. 하지만 2126 <y <2128의 경우 __fdividef(x, y)는 0의 결과를 전달하지만 "/" 연산자는 표 C-4에 설명된 정확도 내에서 올바른 결과를 전달합니다.

Also, for 2126 < y < 2128, if x is infinity, __fdividef(x,y) delivers a NaN (as a result of multiplying infinity by zero), while the “/” operator returns infinity. 
> 또한 2126 <y <2128인 경우 x가 무한대인 경우 __fdividef(x, y)는 NaN (무한대에 0을 곱한 결과)을 전달하지만 "/" 연산자는 무한대를 반환합니다.

On the other hand, the "/" operator is IEEE-compliant on devices of compute capability 2.x and higher when the code is compiled with -prec-div=true or without any -prec-div option at all since its default value is true.  
> 반면에 "/" 연산자는 기본값이 true이므로 -prec-div=true 또는 -prec-div 옵션을 사용하지 않고 코드를 컴파일하면 컴퓨팅 기능이 2.x 이상인 디바이스에서 IEEE 규격을 준수합니다.

Table C-4. Single-Precision Floating-Point Intrinsic Functions Supported by the CUDA Runtime Library with Respective Error Bounds 
> 표 C-4. 각각의 오류 범위와 함께 CUDA 런타임 라이브러리가 지원하는 단 정밀도 부동 소수점 내장 함수

The maximum ulp error is 2 + floor(abs(2.95 * x)). 
> 최대 ulp 오류는 2 + floor (abs(2.95 * x))입니다.

For x in [0.5, 2], the maximum absolute error is 2-21.41, otherwise, the maximum ulp error is 3. 
> [0.5, 2]에서 x의 경우 최대 절대 오류는 2-21.41이고, 그렇지 않으면 최대 ulp 오류는 3입니다. 

For x in [0.5, 2], the maximum absolute error is 2-22, otherwise, the maximum ulp error is 2. 
> [0.5, 2]에서 x의 경우, 최대 절대 오류는 2-22이고, 그렇지 않으면 최대 ulp 오차는 2입니다.

For x in [0.5, 2], the maximum absolute error is 2-24, otherwise, the maximum ulp error is 3.  
> [0.5, 2]에서 x의 경우 최대 절대 오류는 2-24이고, 그렇지 않으면 최대 ulp 오류는 3입니다.

For x in [-, ], the maximum absolute error is 2-21.41, and larger otherwise. 
> [-,]에서 x의 경우 최대 절대 오류는 2-21.41이고 그렇지 않은 경우 더 큽니다.

For x in [-, ], the maximum absolute error is 2-21.19, and larger otherwise. 
> [-,]에서 x의 경우 최대 절대 오류는 2-21.19이며 그렇지 않으면 더 큽니다.

Same as sinf(x) and cosf(x). 
> sinf(x) 및 cosf(x)와 동일합니다. 

Derived from its implementation as exp2f(y * __log2f(x)). 
> exp2f(y * __log2f(x))의 구현에서 파생됩니다.

## C.2.2 Double-Precision Floating-Point Functions 
> C.2.2 배 정밀도 부동 소수점 함수

__dadd_rn() and __dmul_rn() map to addition and multiplication operations that the compiler never merges into FMADs. 
> __dadd_rn() 및 __dmul_rn()은 컴파일러가 FMAD에 병합하지 않는 덧셈 및 곱셈 연산에 매핑됩니다.

By contrast, additions and multiplications generated from the '*' and '+' operators will frequently be combined into FMADs. 
> 대조적으로 '*' 및 '+' 연산자에서 생성된 덧셈 및 곱셈은 자주 FMAD에 결합됩니다.

Table C-5. Double-Precision Floating-Point Intrinsic Functions Supported by the CUDA Runtime Library with Respective Error Bounds 
> 표 C-5. 각각의 오류 범위와 함께 CUDA 런타임 라이브러리가 지원하는 배 정밀도 부동 소수점 내장 함수 

