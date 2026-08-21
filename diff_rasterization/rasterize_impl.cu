#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include  "rasterize_impl.h"
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
#include <cooperative_groups.h>
#include "utils.h"
#include "forward.h"
#include "backward.h"


namespace cg = cooperative_groups;
cudaRasterizer::geometry_state cudaRasterizer::geometry_state::fromChunk(char*& chunk,size_t P)
{
    cudaRasterizer::geometry_state geometry;
    obtain(chunk,geometry.depth,P,128);
    obtain(chunk,geometry.radii,P,128);
    obtain(chunk,geometry.conic,P*3,128);
    obtain(chunk,geometry.cen2p,P*2,128);
    obtain(chunk,geometry.cov3w,P*6,128);
    obtain(chunk,geometry.cov3r,P*6,128);
    obtain(chunk,geometry.cov2r,P*3,128);
    obtain(chunk,geometry.bias,P,128);
    obtain(chunk,geometry.tiles_touched,P,128);
    obtain(chunk,geometry.point_offset,P,128);
	cub::DeviceScan::InclusiveSum(nullptr, geometry.scan_size, geometry.tiles_touched, geometry.tiles_touched, P);
    obtain(chunk,geometry.scan_space,geometry.scan_size,128);
    return geometry;
}


cudaRasterizer::binning_state cudaRasterizer::binning_state::fromChunk(char*& chunk,size_t P)
{
    cudaRasterizer::binning_state binning;
    obtain(chunk,binning.unsorted_key_list,P,128);
    obtain(chunk,binning.key_list,P,128);
    obtain(chunk,binning.unsorted_value_list,P,128);
    obtain(chunk,binning.value_list,P,128);
    cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.unsorted_key_list, binning.key_list,
		binning.unsorted_value_list, binning.value_list, P);
	obtain(chunk, binning.sorting_space, binning.sorting_size, 128);
    return binning;
}


cudaRasterizer::image_state cudaRasterizer::image_state::fromChunk(char*& chunk,size_t P)
{
    cudaRasterizer::image_state image;

    obtain(chunk,image.Tfinal,P,128);
    obtain(chunk,image.ncontributor,P,128);
    obtain(chunk,image.ranges,P,128);
    return image;
}

cudaRasterizer::GradState cudaRasterizer::GradState::fromChunk(char*& chunk, size_t P)
{
	GradState grad;
	obtain(chunk,grad.dl_dconic,P*3,128);
	obtain(chunk,grad.dl_dcov3r,P*6,128);
	obtain(chunk,grad.dl_dcov2r,P*3,128);
	obtain(chunk,grad.dl_dcen2p,P*2,128);
	return grad;
}



__global__ void duplicate_with_keys(
    const uint32_t* point_offset,
    const int num_gaussians,
    const float* radii,
    const float* cen2p,
    const float* depths,
    const dim3 grid,
    uint64_t* unsorted_key_list,
    uint64_t* unsorted_value_list

)
{
    auto idx = cg::this_grid().thread_rank();
    if(idx>=num_gaussians){return;}
    int start = (idx==0) ? 0 : point_offset[idx-1];
    uint2 rctmin;
    uint2 rctmax;
    getrect(cen2p+idx*2,radii[idx],rctmin,rctmax,grid);
    float depth = depths[idx];
    for(int i =rctmin.x;i<rctmax.x;i++)
    {
        for(int j=rctmin.y;j<rctmax.y;j++)
        {
            uint64_t tile_id = j*grid.x+i;
            
            unsorted_key_list[start] =
            ((tile_id) << 32) |*((uint32_t*)&depth);
            unsorted_value_list[start]  = idx;
            start++;
        }
    }
}


__global__ void IdentifyRanges(
    uint2* ranges,
    const  uint64_t* key_list,
    const  int num_rendered
)
{
    auto idx = cg::this_grid().thread_rank();
    if(idx>=num_rendered){return;}
    uint32_t pretile = (idx==0) ? 0 : key_list[idx-1]>>32;
    if(idx==0){ranges[pretile].x=0;}
    if(idx==num_rendered-1){ranges[pretile].y = num_rendered;}

    uint32_t curtile = key_list[idx]>>32;
    if(pretile!=curtile)
    {
        ranges[curtile].x = idx;
        ranges[pretile].y = idx;
    }
    
}







int cudaRasterizer::forward(
    uint32_t N,
    uint32_t height,
    uint32_t width,
    float fx,
    float fy,    
    float* viewmatrix,
    float* projmatrix,    
    float* q,
    float* s,

    float*  mean3d,
    float*  opacity,
    float*  colors,
    float*  image,
    dim3 grid,
    dim3 block,
    std::function<char*(size_t N)> imageFunc,
    std::function<char*(size_t N)> geomFunc,
    std::function<char*(size_t N)> binnFunc 
  )
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_ms = 0.0f;

    size_t geostate_size = required<cudaRasterizer::geometry_state>(N);
    char* geo_chunk = geomFunc(geostate_size);
    cudaRasterizer::geometry_state geostate =  cudaRasterizer::geometry_state::fromChunk(geo_chunk ,N);

    size_t imgstate_size = required<cudaRasterizer::image_state>(width*height);
    char* chunk_img = imageFunc(imgstate_size);
    cudaRasterizer::image_state imgstate = cudaRasterizer::image_state ::fromChunk(chunk_img,width*height); 

    int debug=1;
    cudaEventRecord(start);
    CHECK_CUDA(Forward::preprocess(
        q,
        s,
        mean3d,
        viewmatrix,
        projmatrix,
        N,
        height,
        width,
        fx,
        fy,
        grid,
        geostate.conic,
        geostate.depth,
        geostate.radii,
        geostate.tiles_touched,
        geostate.cov3w,
        geostate.cen2p,
        geostate.bias
    ),debug)
    cudaEventRecord(stop);
// 等待 stop 事件真正执行完成
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    printf("Forward::preprocess Kernel time: %.3f ms\n", elapsed_ms);

    CHECK_CUDA(cub::DeviceScan::InclusiveSum(geostate.scan_space, geostate.scan_size, geostate.tiles_touched, geostate.point_offset, N),debug);
    int num_rendered;
    CHECK_CUDA(cudaMemcpy(&num_rendered,geostate.point_offset+N-1,sizeof(int),cudaMemcpyDeviceToHost),debug);

    size_t binning_size = required<cudaRasterizer::binning_state>(num_rendered);
    char* chunk_binn = binnFunc(binning_size);
    cudaRasterizer::binning_state binstate =cudaRasterizer::binning_state::fromChunk(chunk_binn,num_rendered); 
    duplicate_with_keys<<<(N+255)/256,256>>>(
            geostate.point_offset,
            N,
            geostate.radii,
            geostate.cen2p,

            geostate.depth,
            grid,
            binstate.unsorted_key_list,
            binstate.unsorted_value_list
            );
    CHECK_CUDA(,debug)

    cudaEventRecord(start);
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(binstate.sorting_space,binstate.sorting_size,
                                    binstate.unsorted_key_list,binstate.key_list,
                                    binstate.unsorted_value_list,binstate.value_list,num_rendered),debug);
    cudaEventRecord(stop);
// 等待 stop 事件真正执行完成
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    printf("RadixSort Kernel time: %.3f ms\n", elapsed_ms);
    printf("num_gaussians: %d, num_rendered: %d, resolution: %d X %d\n", N,num_rendered,height,width);
    CHECK_CUDA(cudaMemset(imgstate.ranges, 0, grid.x * grid.y * sizeof(uint2)),debug);
    if(num_rendered>0)
    {
        IdentifyRanges<<<(num_rendered+255)/256,256>>>(
                imgstate.ranges,
                binstate.key_list,
                num_rendered
            );            
    }
    CHECK_CUDA(,debug)   



    cudaEventRecord(start);
    CHECK_CUDA(Forward::render(
        height,
        width,
        geostate.conic,
        geostate.cen2p,
        geostate.bias,
        colors,
        opacity,
        imgstate.ranges,
        binstate.value_list,
        grid,
        block,  
        imgstate.Tfinal,
        imgstate.ncontributor,
        image
    ),debug)

    cudaEventRecord(stop);
// 等待 stop 事件真正执行完成
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed_ms, start, stop);
    printf("Forward::render Kernel time: %.3f ms\n", elapsed_ms);


    return num_rendered;

}


void cudaRasterizer::backward(
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
    std::function<char*(size_t N)> gradFunc 
)
{
    int debug = 1;
    size_t grad_size = required<cudaRasterizer::GradState>(N);
    char* grad_chunk = gradFunc(grad_size);
    cudaRasterizer::GradState gradstate     = cudaRasterizer::GradState::fromChunk(grad_chunk,N);
    cudaRasterizer::geometry_state geostate = cudaRasterizer::geometry_state::fromChunk(geometry_buffer,N);
    cudaRasterizer::binning_state binnstate = cudaRasterizer::binning_state::fromChunk(binning_buffer,num_rendered); 
    cudaRasterizer::image_state imgstate    = cudaRasterizer::image_state::fromChunk(image_buffer,image_width*image_height); 

    CHECK_CUDA(Backward::render(
        image_height,
        image_width,
        dl_dpixel,                  //height*width*chan
        geostate.conic,
        geostate.cen2p,
        geostate.bias,
        colors,
        opacity,
        imgstate.Tfinal,
        imgstate.ncontributor,
        geostate.cov3r,
        geostate.cov2r,
        imgstate.ranges,
        binnstate.value_list,
        grid,
        block,   
        dl_dcolor,
        dl_dopacity,
        gradstate.dl_dconic,
        gradstate.dl_dcov3r, 
        gradstate.dl_dcov2r,
        gradstate.dl_dcen2p),debug);

    CHECK_CUDA(Backward::preprocess(
        N,
        fx,
        fy,
        viewmatrix,
        projmatrix,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
        cen3w,
        geostate.cov3w,
        geostate.conic,
      
        q,
        s,
      
        gradstate.dl_dconic,
        gradstate.dl_dcov3r,
        gradstate.dl_dcov2r,
        gradstate.dl_dcen2p,
      
        dl_dq,
        dl_ds,
        dl_dcen3w),debug);
}