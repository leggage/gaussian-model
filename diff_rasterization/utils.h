


#include "stdio.h"


__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)  //matrix 按列优先顺序存储
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14], 
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)  //matrix 按列优先顺序存储
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14], 
	};
	return transformed;
}
__forceinline__ __device__ int ndc2pixel(float p,int len)
{
    int pixel = (p+1)*len/2;
    return pixel;
}