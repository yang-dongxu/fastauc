from setuptools import setup, Extension, find_packages
import numpy

setup(
    name='fastauc',
    version='0.1.0',
    description='A package to calculate AUROC and AUPR using a C++ extension',
    author='dxyang',
    author_email='yang_dongxu@qq.com',
    url='https://github.com/yang-dongxu/fastauc',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    ext_modules=[
        Extension(
            name='fastauc.cpp_auc',  # This should match the import path for the compiled module
            sources=[os.path.join('src','cpp_auc.cpp')],
            include_dirs=[numpy.get_include()],
            extra_compile_args=['-shared', '-O3', '-march=native'],
        ),
    ],
    install_requires=[
        'numpy',
        'pathlib'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
