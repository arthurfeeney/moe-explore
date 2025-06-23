# peak metrics are flop/s and byte/s
PEAK_METRICS=sm__ops_path_tensor_src_fp16_dst_fp32_sparsity_off.sum.peak_sustained,sm__cycles_elapsed.avg.per_second,dram__bytes.sum.peak_sustained,dram__cycles_elapsed.avg.per_second,device__attribute_max_gpu_frequency_khz

# used to compute flops and arithmetic intensity
ACHIEVED_METRICS=sm__ops_path_tensor_src_fp16_dst_fp32_sparsity_off.sum.per_cycle_elapsed,sm__cycles_elapsed.avg.per_second,dram__bytes.sum.per_second

ncu \
        --profile-from-start off \
        --export profile \
        --metrics ${PEAK_METRICS},${ACHIEVED_METRICS} \
        -f \
        python profile_methods.py