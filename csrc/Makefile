LIBS=-I/pub/afeeney/project/cutlass/include
CUDA_ARGS=-arch=compute_80 -code=sm_80
target:
	nvcc -std=c++17 $(LIBS) $(CUDA_ARGS) cutlass_group_gemm.cu --expt-relaxed-constexpr
