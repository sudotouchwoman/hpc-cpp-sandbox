module purge

module load\
    INTEL/oneAPI_2022_env\
    cpp_tools/boost/v1.88.0\
    OpenBlas/v0.3.23-gnu12-intel\
    cmake/3.31.8\
    nvidia_sdk/nvhpc/23.5
    # gnu14/14.1\
    # OpenBlas/v0.3.18\
    # tools/autotools/v2.72
    # llvm/v13.0.1/clang\

export Boost_ROOT=/opt/hse/libs/cpp_tools/boost/v1.88.0
export MKL_ROOT=/opt/software/intel/oneapi_2022/mkl/latest

module list
