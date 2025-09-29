module purge

module load\
    INTEL/oneAPI_2022_env\
    cpp_tools/boost/v1.88.0\
    OpenBlas/v0.3.23-gnu12-intel\
    cmake/3.21.3
    # gnu14/14.1\
    # OpenBlas/v0.3.18\
    # tools/autotools/v2.72
    # llvm/v13.0.1/clang\

export Boost_ROOT=/opt/hse/libs/cpp_tools/boost/v1.88.0

module list
