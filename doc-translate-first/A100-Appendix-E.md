## Appendix E. Texture Fetching 
> 부록 E. 텍스처 가져오기

This appendix gives the formula used to compute the value returned by the texture functions of Section B.8 depending on the various attributes of the texture reference (see Section 3.2.10). 
> 이 부록은 텍스처 참조의 다양한 속성에 따라 B.8 절의 텍스처 함수에 의해 반환된 값을 계산하는 데 사용되는 공식을 제공합니다 (3.2.10 절 참조).

The texture bound to the texture reference is represented as an array T of  N texels for a one-dimensional texture,  M N texels for a two-dimensional texture,  L MN   texels for a three-dimensional texture. 
> 텍스처 참조에 바인딩된 텍스처는 1 차원 텍스처에 대한 N 텍셀의 배열 T로 표현되며, 2 차원 텍스처에 대한 M N 텍셀, 3 차원 텍스처에 대한 L MN 텍셀로 표현됩니다.

It is fetched using non-normalized texture coordinates x, y , and z , or the normalized texture coordinates N x/ , M y/ , and L z/ as described in Section 3.2.10.1.2. 
> 3.2.10.1.2 절에 설명된 것처럼 정규화되지 않은 텍스처 좌표 x, y 및 z 또는 정규화된 텍스처 좌표 N x/, M y/ 및 L z/를 사용하여 가져옵니다.

In this appendix, the coordinates are assumed to be in the valid range. 
> 이 부록에서는 좌표가 유효한 범위에 있다고 가정합니다.

Section 3.2.10.1.2 explained how out-of-range coordinates are remapped to the valid range based on the addressing mode. 
> 3.2.10.1.2 절에서 범위를 벗어나는 좌표가 주소 지정 모드를 기반으로 유효한 범위에 재매핑되는 방법을 설명했습니다.

## E.1 Nearest-Point Sampling 
> E.1 가장 가까운 지점 샘플링

In this filtering mode, the value returned by the texture fetch is  ] [)( i Txtex  for a one-dimensional texture,  ] ,[),( j iTyxtex  for a two-dimensional texture,  ] ,,[),,( k jiTzyxtex  for a three-dimensional texture, where ) (xfloori  , ) (yfloorj  , and ) (zfloork. 
> 이 필터링 모드에서 텍스처 가져오기에 의해 반환된 값은 (1 차원 텍스처에 대한 [Txtex]), ([2 차원 텍스처에 대한 iTexxtex,]), ([3 차원 텍스쳐의 경우 k jiTexxtex]), (xfloori,) (yfloorj, and) (zfloork.

Figure E-1 illustrates nearest-point sampling for a one-dimensional texture with 4N . 
> 그림 E-1은 4N이 있는 1 차원 텍스처에 대한 가장 가까운 지점 샘플링을 보여줍니다. 

For integer textures, the value returned by the texture fetch can be optionally remapped to [0.0, 1.0] (see Section 3.2.10.1.1). 
> 정수 텍스처의 경우, 텍스처 가져오기에 의해 반환된 값은 선택적으로 [0.0, 1.0]으로 재매핑될 수 있습니다 (3.2.10.1.1 절 참조). 
 
Figure E-1. Nearest-Point Sampling of a One-Dimensional Texture of Four Texels 
> 그림 E-1. 4 텍셀의 1 차원 텍스처의 가장 가까운 지점 샘플링

## E.2 Linear Filtering 
> E.2 선형 필터링

In this filtering mode, which is only available for floating-point textures, the value returned by the texture fetch is  ] 1[][)1()(  iTiTxtex  for a one-dimensional texture,>  부동 소수점 텍스처에만 사용할 수 있는 이 필터링 모드에서 텍스처 가져오기에서 반환되는 값은 다음과 같습니다. 1 차원 텍스처의 경우 1[][]1()( iTiTxtex,
 
for a three-dimensional texture, where:  ) ( B xfloori  , ) ( B xfrac  , 5 .0 xx B ,  ) ( B yfloorj  , ) ( B yfrac  , 5 .0 yy B ,  ) ( B zfloork  , ) ( B zfrac  , 5 .0 zz B .  ,  , and  are stored in 9-bit fixed point format with 8 bits of fractional value (so 1.0 is exactly represented). 
> 3 차원 텍스처의 경우:) (B xfloori,) (B xfloori, 5 .0 xx B) (B yfloorj) (B yfrac, 5 .0 yy B) (B zfloork) (B zfrac, 5 .0 zz B. 8 비트의 분수 값을 가진 9 비트 고정 소수점 형식으로 저장됩니다 (따라서 1.0은 정확하게 표현됩니다).

Figure E-2 illustrates nearest-point sampling for a one-dimensional texture with 4N . 
> 그림 E-2는 4N의 1 차원 텍스처에 대한 가장 가까운 지점 샘플링을 보여줍니다.
 
Figure E-2. Linear Filtering of a One-Dimensional Texture of Four Texels in Clamp Addressing Mode 
> 그림 E-2. 클램프 주소 지정 모드에서 4 개 텍셀의 1 차원 텍스처의 선형 필터링
 
## E.3 Table Lookup 
> E.3 테이블 룩업(검색)

Figure E-3 illustrates the use of texture filtering to implement a table lookup with 4R or 1 R from a one-dimensional texture with . 
> 그림 E-3은 텍스처 필터링을 사용하여 1 차원 텍스처에서 4R 또는 1R로 테이블 검색을 구현하는 방법을 보여줍니다. 
 
Figure E-3. One-Dimensional Table Lookup Using Linear Filtering 
> 그림 E-3. 선형 필터링을 사용한 1 차원 테이블 검색
