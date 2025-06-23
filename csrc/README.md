# CUTLASS Grouped GEMM

This is some simple experiments with CUTLASS' Grouped GEMM APIs.

`cutlass2_group_gemm.cu` actually uses some of cutlass' 2.0 API. 
The 2.0 API does not use cute and some stuff is not receiving updates.
I.e., cutlass::gemm::threadblock::* is not really maintained.
