r"""
This can be run on an the results of an ncu profile of `profile_gemm.py`.
This prints out the peak TFLOPs, bandwidth, and arithmetic intensity.
These can be used to make a roofline plot.

The matrix size from `profile_gemm.py` should be fairly large, so the 
average cycles per second is accurate. 
"""

import sys
sys.path.append('/opt/apps/cuda/12.2.0/nsight-compute-2023.2.0/extras/python/')
import ncu_report

import numpy as np

float16_metrics = {
    'sm_flops_fp16': 'sm__ops_path_tensor_src_fp16_dst_fp16_sparsity_off.sum.peak_sustained',
    # PyTorch accumulates fp16 gemms in fp32.
    'sm_flops': 'sm__ops_path_tensor_src_fp16_dst_fp32_sparsity_off.sum.peak_sustained',
    'max_kilohertz': 'device__attribute_max_gpu_frequency_khz',
    'sm_cycles': 'sm__cycles_elapsed.avg.per_second',
    'dram_bytes': 'dram__bytes.sum.peak_sustained',
    'dram_cycles': 'dram__cycles_elapsed.avg.per_second'
}

def peak(kernel, metrics):
    # The peak_sustained metrics are hard-coded per GPU and do not depend on the work being done.
    # However, the average cycles per second CAN change depending on the GPU's clock frequency.
    # This means the peak_work and peak_bandwidth may change on different runs if the clock frequency changes.
    peak_work = (
        # (total-across-all-SMs / cycle) * (cycles / second)
        kernel[metrics['sm_flops']].value() *
        kernel[metrics['max_kilohertz']].value() * 1000
    )
    peak_bandwidth = (
        # (bytes accessed / cycle) * (cycles / second)
        kernel[metrics['dram_bytes']].value() *
        kernel[metrics['dram_cycles']].value()
    )
    return peak_work, peak_bandwidth

def print_kernel_metrics(kernel, metrics):
    for key in metrics.values():
        m = kernel[key].value()
        print(f'Metric[{key}]: {m}')

def get_data(kernels, func, metrics):
    data = [func(kernel, metrics) for kernel in kernels]
    flops_per_sec, bytes_per_sec = zip(*data)
    flops_per_byte = [f / b for (f, b) in data]
    return np.array(flops_per_sec), np.array(flops_per_byte), np.array(bytes_per_sec)

if __name__ == "__main__":
    context = ncu_report.load_report('profile.ncu-rep')
    kernels = context[0]
    
    # An A30 GPU has 56 SMs, each SM has 4 subpartitions with tensor cores. So, 224 tensor cores
    # Then, there should be 224 * 512 == 114688 flops per cycle. 
    print('Kernel Metrics:')
    for k in kernels:
        print_kernel_metrics(k, float16_metrics)
    
    flops_per_sec, flops_per_byte, bytes_per_sec = get_data(kernels, peak, float16_metrics)
    
    tflops_per_sec = 1e-12 * flops_per_sec
    tbytes_per_sec = 1e-12 * bytes_per_sec
    
    print(f'Peak TFLOPS/s {tflops_per_sec}')
    print(f'Peak TB/s {tbytes_per_sec}')
    print(f'Machine Balance: FLOPS/Byte {flops_per_byte}')