from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = Path(__file__).resolve().parent

setup(
    name='rasterization',
    ext_modules=[
        CUDAExtension(
                name='cuda_rasterization',
                sources=['ext.cpp', 'forward.cu','backward.cu','rasterize_impl.cu','rasterize_point.cu'],
                include_dirs=[str(this_dir / 'third_party')],
                extra_compile_args={'cxx': ['-g'],
                                    'nvcc': ['-O2']},
                extra_link_args=['-Wl,--no-as-needed', '-lcuda'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
