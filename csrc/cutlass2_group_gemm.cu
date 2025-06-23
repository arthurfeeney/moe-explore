/**
 * Based on pytorch's NestedTensor matmul, but tries to remove the pytorch internal stuff.
 * It's basically c++ + cutlass + a simple pytorch extension wrapping it.
 */
#include <iostream>

#include <cuda_fp16.h>
#include <cute/layout.hpp>
#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template<
  typename scalar_t,
  unsigned int kPad,
  typename LayoutA,
  typename LayoutB,
  typename OpClass,
  typename Arch,
  typename ThreadBlockShape,
  typename WarpShape,
  typename InstructionShape  
>
void cutlass_group_gemm(
  // a, b, c are vectors of device pointers
  const std::vector<scalar_t*> a,
  const std::vector<scalar_t*> b,
  const std::vector<scalar_t*> c,
  const std::vector<int>& lda,
  const std::vector<int>& ldb,
  const std::vector<int>& ldd,
  const std::vector<cutlass::gemm::GemmCoord>& gemm_sizes,
  const int problem_count
) {
  using Element = scalar_t;
  using ElementAcc = float;

  using GemmConfiguration =
      typename cutlass::gemm::device::DefaultGemmConfiguration<
          OpClass,
          Arch,
          Element,
          Element,
          Element,
          ElementAcc>;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
      Element,
      LayoutA,
      cutlass::ComplexTransform::kNone,
      kPad,
      Element,
      LayoutB,
      cutlass::ComplexTransform::kNone,
      kPad,
      Element,
      cutlass::layout::RowMajor,
      ElementAcc,
      OpClass,
      Arch,
      ThreadBlockShape,
      WarpShape,
      InstructionShape,
      typename GemmConfiguration::EpilogueOutputOp,
      // I believe cutlass group gemm does not support swizzling.
      // This parameter seems to be unused.
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
      GemmConfiguration::kStages>::GemmKernel;

  using GemmGrouped = typename cutlass::gemm::device::GemmGrouped<GemmKernel>;
  using EpilogueOutputOp = typename GemmGrouped::GemmKernel::Epilogue::OutputOp;
  typename EpilogueOutputOp::Params epilogue_op(/*alpha*/ 1, /*beta*/ 0);

  const int64_t gemm_coord_size =
    problem_count * ((int64_t)sizeof(cutlass::gemm::GemmCoord));
  
  thrust::host_vector<int64_t> host_gmm_args(problem_count * 6 + gemm_coord_size);

  // Obtain pointers for each argument (on host)
  int64_t* host_lda_data = host_gmm_args.data(); // Base pointer
  int64_t* host_ldb_data = host_lda_data + problem_count;
  int64_t* host_ldd_data = host_lda_data + 2 * problem_count;
  int64_t* host_ptr_a_data = host_lda_data + 3 * problem_count;
  int64_t* host_ptr_b_data = host_lda_data + 4 * problem_count;
  int64_t* host_ptr_c_data = host_lda_data + 5 * problem_count;
  cutlass::gemm::GemmCoord* host_problem_sizes_data =
      reinterpret_cast<cutlass::gemm::GemmCoord*>(host_lda_data + 6 * problem_count);

  // Set arguments into gmm_args from input args
  for (int i = 0; i < problem_count; ++i) {
    host_problem_sizes_data[i] = gemm_sizes[i];
    host_lda_data[i] = lda[i];
    host_ldb_data[i] = ldb[i];
    host_ldd_data[i] = ldd[i];
    host_ptr_a_data[i] = reinterpret_cast<int64_t>(a[i]);
    host_ptr_b_data[i] = reinterpret_cast<int64_t>(b[i]);
    host_ptr_c_data[i] = reinterpret_cast<int64_t>(c[i]);
  }
  const int threadblock_count =
      GemmGrouped::sufficient(host_problem_sizes_data, problem_count);

  // Transfer arguments to GPU
  thrust::device_vector<int64_t> gmm_args = host_gmm_args;

  // Obtain pointers for each of arguments (on GPU)
  int64_t* lda_data = thrust::raw_pointer_cast(gmm_args.data()); // Base pointer
  int64_t* ldb_data = lda_data + problem_count;
  int64_t* ldd_data = lda_data + 2 * problem_count;
  int64_t* ptr_a_data = lda_data + 3 * problem_count;
  int64_t* ptr_b_data = lda_data + 4 * problem_count;
  int64_t* ptr_c_data = lda_data + 5 * problem_count;
  cutlass::gemm::GemmCoord* problem_sizes_data =
      reinterpret_cast<cutlass::gemm::GemmCoord*>(lda_data + 6 * problem_count);

  // Create GemmGrouped::Arguments using the arguments prepared above
  typename GemmGrouped::Arguments args(
      problem_sizes_data,
      problem_count,
      threadblock_count,
      epilogue_op,
      reinterpret_cast<Element**>(ptr_a_data),
      reinterpret_cast<Element**>(ptr_b_data),
      reinterpret_cast<Element**>(ptr_c_data),
      reinterpret_cast<Element**>(ptr_c_data),
      lda_data,
      ldb_data,
      ldd_data,
      ldd_data);

  GemmGrouped gemm;
  cutlass::Status status =
      gemm.initialize(args, nullptr);

  gemm.run();
}

template<
  typename ThreadBlockShape,
  typename WarpShape,
  typename InstructionShape,
  typename ...Args>
void rowmajor_sm80_half_group_gemm(Args... args) { 
  // alignment... __half has to be aligned to multiples of 8 for tensor cores
  // technically, as long as m / n / k are all multiples of 8, it's fine, so
  // just setting kPad to 8.
  constexpr int kPad = 8;

  return cutlass_group_gemm<
    cutlass::half_t,
    kPad,
    /*LayoutA*/ cutlass::layout::RowMajor,
    /*LayoutB*/ cutlass::layout::RowMajor,
    /*OpClass*/ cutlass::arch::OpClassTensorOp,
    /*arch*/ cutlass::arch::Sm80,
    ThreadBlockShape,
    WarpShape,
    InstructionShape>(args...);
}

int main() {

  int problem_count = 10;

  using scalar_t = cutlass::half_t;

  std::vector<cutlass::gemm::GemmCoord> gemm_sizes(problem_count);
  std::vector<int> lda(problem_count);
  std::vector<int> ldb(problem_count);
  std::vector<int> ldc(problem_count);
  std::vector<thrust::device_vector<scalar_t>> a(problem_count);
  std::vector<thrust::device_vector<scalar_t>> b(problem_count);
  std::vector<thrust::device_vector<scalar_t>> c(problem_count);

  for (int i = 0; i < problem_count; ++i) {
    int m = 128;
    int n = 128;
    int k = 128;
    gemm_sizes.at(i) = cutlass::gemm::GemmCoord(m, n, k);

    // assuming row major matrices
    lda.at(i) = k;
    ldb.at(i) = n;
    ldc.at(i) = n;

    a.at(i) = thrust::device_vector<scalar_t>(m * k);
    b.at(i) = thrust::device_vector<scalar_t>(k * n);
    c.at(i) = thrust::device_vector<scalar_t>(m * n);

    thrust::fill(a.at(i).begin(), a.at(i).end(), cutlass::half_t(1.0));
    thrust::fill(b.at(i).begin(), b.at(i).end(), cutlass::half_t(1.0));
    thrust::fill(c.at(i).begin(), c.at(i).end(), cutlass::half_t(0.0));
  }

  std::vector<scalar_t*> a_ptrs(problem_count);
  std::vector<scalar_t*> b_ptrs(problem_count);
  std::vector<scalar_t*> c_ptrs(problem_count);

  for (int i = 0; i < problem_count; ++i) {
    a_ptrs.at(i) = thrust::raw_pointer_cast(a.at(i).data());
    b_ptrs.at(i) = thrust::raw_pointer_cast(b.at(i).data());
    c_ptrs.at(i) = thrust::raw_pointer_cast(c.at(i).data());
  }

  rowmajor_sm80_half_group_gemm<
    cutlass::gemm::GemmShape<128, 128, 128>, 
    cutlass::gemm::GemmShape<64, 64, 64>, 
    cutlass::gemm::GemmShape<16, 8, 16>
  >(
    a_ptrs,
    b_ptrs,
    c_ptrs,
    lda,
    ldb,
    ldc,
    gemm_sizes,
    problem_count
  );

  return 0;
}
