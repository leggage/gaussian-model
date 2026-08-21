#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <fstream>
#include <string>
#include <functional>

namespace cudaRasterizer{
    template<typename T>
    static void obtain(char*& chunk,T*& ptr,size_t P,size_t alignment)
    {
        size_t offset = reinterpret_cast<uintptr_t>(chunk+alignment-1) &~ (alignment-1);
        ptr = reinterpret_cast<T*>(offset);
        chunk = reinterpret_cast<char*>(ptr+P);
    }

    struct geometry_state{
        float* depth;
        float* radii;
        float* conic;
        float* cen2p;
        float* cov3w;
        float* cov3r;
        float* cov2r; 
        float* bias;
        uint32_t* tiles_touched;
        uint32_t* point_offset;
        size_t scan_size;
        char* scan_space;
        static geometry_state fromChunk(char*& chunk,size_t N);
    };

    struct binning_state{
        uint64_t* unsorted_key_list;
        uint64_t* key_list;
        uint64_t* unsorted_value_list;
        uint64_t* value_list;
        size_t sorting_size;
        char* sorting_space;
        static binning_state fromChunk(char*& chunk,size_t N);
    };

    struct image_state{
        float* Tfinal;
        uint32_t*  ncontributor;
        uint2* ranges;
        static image_state fromChunk(char*& chunk,size_t N);
    };

	struct GradState
	{
		float* dl_dconic;
		float* dl_dcov3r;
		float* dl_dcov2r;
		float* dl_dcen2p;
		static GradState fromChunk(char*& chunk, size_t P);
	};

    int forward(
        uint32_t N,
        uint32_t height,
        uint32_t width,
        float fx,
        float fy,    
        float* viewmatrix,
        float* projmatrix,    
        float* q,
        float* s,
    
        float*  mean3w,
        float*  opacity,
        float*  colors,
        float*  image,
        dim3 grid,
        dim3 block,
        std::function<char*(size_t N)> imageFunc,
        std::function<char*(size_t N)> geomFunc,
        std::function<char*(size_t N)> binnFunc);

    void backward(
        const int N,
        const int num_rendered, 
        const dim3 grid,
        const dim3 block,

        const uint32_t image_height,
        const uint32_t image_width,
        const float fx,
        const float fy,
        const float* viewmatrix,
        const float* projmatrix,

        char* image_buffer,
        char* binning_buffer,
        char* geometry_buffer,  

        const float* dl_dpixel,
        const float* colors,
        const float* opacity, 
        const float* cen3w,
        const float* q,
        const float* s,

        float* dl_dcolor,
        float* dl_dopacity,
        float* dl_dq,
        float* dl_ds,
        float* dl_dcen3w,
        std::function<char*(size_t N)> gradFunc);
    
    template<typename T>
    size_t required(size_t N){
        char* size = nullptr;
        T::fromChunk(size,N);
        return reinterpret_cast<size_t>(size+128);
    }

};
