

namespace Forward{
    void render(
        const int image_height,
        const int image_width,
        const float* __restrict__ conics2d,
        const float* __restrict__ means2d,
        const float* __restrict__ weights,
        const float* __restrict__ colors,
        const float* __restrict__ opacity,
        const uint2* __restrict__ ranges,
        const uint64_t* __restrict__ point_list,
        const dim3 grid,
        const dim3 block,  
        float* __restrict__ Tfinal,
        uint32_t* __restrict__ ncontributor,
        float* __restrict__ image
      );

    void preprocess(
    const float* __restrict__ q,
    const float* __restrict__ s,
    const float* __restrict__ mean3d,
    const float* __restrict__ viewmatrix,
    const float* __restrict__ projmatrix,
    const int N,
    const int height,
    const int width,
    const float fx,
    const float fy,
    const dim3 grid,
    float* __restrict__ conics2d,
    float* __restrict__ depths,
    float* __restrict__ radii,
    uint32_t* __restrict__ tiles_touched, 
    float* __restrict__ cov3w,
    float* __restrict__ means2d,
    float* __restrict__ weights  
    );
};