


#include "stdio.h"
#include "config.h"

__forceinline__ __device__ float4 transformPoint4x4_Mcol(const float3& p, const float* matrix)  //matrix 按列优先顺序存储
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14], 
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformPoint4x3_Mcol(const float3& p, const float* matrix)  //matrix 按列优先顺序存储
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14], 
	};
	return transformed;
}

__forceinline__ __device__ float3 transformPoint4x3_Mrow(const float3& p, const float* matrix)  //matrix 按行优先顺序存储
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z + matrix[3],
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z + matrix[7],
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z + matrix[11], 
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4_Mrow(const float3& p, const float* matrix)  //matrix 按行优先顺序存储
{
	float4 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z + matrix[3],
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z + matrix[7],
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z + matrix[11], 
        matrix[12] * p.x + matrix[13] * p.y + matrix[14] * p.z + matrix[15], 
	};
	return transformed;
}


__forceinline__ __device__ void getrect(const float* cen2p,const float radii,uint2& rctmin,uint2& rctmax,dim3 grid)
{
    rctmin={
    min(grid.x,max((int)0,(int)(cen2p[0]-radii)/BlockSize_x)),
    min(grid.y,max((int)0,(int)(cen2p[1]-radii)/BlockSize_y))
    };
    rctmax={
    min(grid.x,max((int)0,(int)(cen2p[0]+radii)/BlockSize_x)),
    min(grid.y,max((int)0,(int)(cen2p[1]+radii)/BlockSize_y))
    };
}

//near-plane culling
__forceinline__ __device__ bool checkinfrustum(const float3& cen3w,const float* viewmatrix,float3& cen3c)
{
    cen3c = transformPoint4x3_Mrow(cen3w,viewmatrix);
    if(cen3c.z<0.2)
    {
        return false;
    }
    return true;
}


//[-1,1]--->[-0.5,width-0.5]or[-0.5,height-0.5]
__forceinline__ __device__ float ndc2pixel(float p, int len)
{
	    return ((p + 1.0f) * static_cast<float>(len)-1) / 2.0f;
}

//coumpute rotation from q
__forceinline__ __device__ void computeRotation(const float4& q, float* R)
{
    // q: (qr, qi, qj, qk) = (w, x, y, z)
    float qr = q.x;
    float qi = q.y;
    float qj = q.z;
    float qk = q.w;

    // 可选：若 q 未归一化，先 normalize（3DGS 通常在调用前处理）
    // float inv_len = rsqrtf(qr*qr + qi*qi + qj*qj + qk*qk);
    // qr *= inv_len; qi *= inv_len; qj *= inv_len; qk *= inv_len;

    R[0] = 1.0f - 2.0f * (qj * qj + qk * qk);
    R[1] = 2.0f * (qi * qj - qr * qk);
    R[2] = 2.0f * (qi * qk + qr * qj);

    R[3] = 2.0f * (qi * qj + qr * qk);
    R[4] = 1.0f - 2.0f * (qi * qi + qk * qk);
    R[5] = 2.0f * (qj * qk - qr * qi);

    R[6] = 2.0f * (qi * qk - qr * qj);
    R[7] = 2.0f * (qj * qk + qr * qi);
    R[8] = 1.0f - 2.0f * (qi * qi + qj * qj);
}
