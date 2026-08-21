


namespace Backward{
    
    void render(
        const int image_height,
        const int image_width,
        const float* __restrict__ dl_dpixel, //height*width*chan
        const float* __restrict__ conics2d,
        const float* __restrict__ means2d,
        const float* __restrict__ weights,
        const float* __restrict__ colors,
        const float* __restrict__ opacity,
        const float* __restrict__ Tfinal,
        const uint32_t* __restrict__ ncontributor,
        const float* __restrict__ cov3r,
        const float* __restrict__ cov2r,
        const uint2* __restrict__ ranges,
        const uint64_t* __restrict__ point_list,
        const dim3 grid,
        const dim3 block,   
        float* __restrict__ dl_dcolor,
        float* __restrict__ dl_dopacity,
        float* __restrict__ dl_dconics2d,
        float* __restrict__ dl_dcovr3d, 
        float* __restrict__ dl_dcovr2d,
        float* __restrict__ dl_dcenp2d 
    );
 
    void preprocess(
        const int gaussian_num,
        const float fx,
        const float fy,
        const float* __restrict__ viewmatrix,
        const float* __restrict__ projmatrix,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
        const float* __restrict__ cen3w,
        const float* __restrict__ cov3w,
        const float* __restrict__ conics2d,
      
        const float* __restrict__ q,
        const float* __restrict__ s,
      
        const float* __restrict__ dl_dconics2d,
        const float* __restrict__ dl_dcovr3d,
        const float* __restrict__ dl_dcovr2d,
        const float* __restrict__ dl_dcenp2d,
      
        float* __restrict__ dl_dq,
        float* __restrict__ dl_ds,
        float* __restrict__ dL_dcen3w
      );
}