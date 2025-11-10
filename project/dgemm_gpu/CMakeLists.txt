# DGEMM GPU Library CMakeLists.txt

# Check if CUDA is available
if(CUDAToolkit_FOUND)
  # Create library target
  add_library(dgemm_gpu STATIC)

  # Set source files
  target_sources(dgemm_gpu PRIVATE
      src/basic.cu
      src/unified.cu
  )

  # Set include directories - PUBLIC makes them available to
  # targets that link to dgemm_gpu
  target_include_directories(dgemm_gpu PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  # Set CUDA language properties
  set_target_properties(dgemm_gpu PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )

  # Link against CUDA runtime
  target_link_libraries(dgemm_gpu PUBLIC CUDA::cudart)

  # Add compile definition to indicate CUDA is available
  target_compile_definitions(dgemm_gpu PUBLIC HAVE_CUDA)
else()
  message(STATUS "CUDA toolkit not found - dgemm_gpu library will not be built")
endif()

