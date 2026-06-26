from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(
    name='rasterization',
    ext_modules=[
        CUDAExtension(
                name='cuda_rasterization',
                sources=['ext.cpp', 'forward.cu','backward.cu'],
                extra_compile_args={'cxx': ['-g'],
                                    'nvcc': ['-O2']},
                extra_link_args=['-Wl,--no-as-needed', '-lcuda'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })