import sys
sys.path.append('/opt/apps/cuda/12.2.0/nsight-compute-2023.2.0/extras/python/')
import ncu_report

import matplotlib.pyplot as plt
import numpy as np
import torch

context = ncu_report.load_report('profile.ncu-rep')
kernel = context[0][0]

def acheived(kernel):
    achieved_work = (
        kernel['sm__ops_path_tensor_src_fp16_dst_fp32_sparsity_off.sum.per_cycle_elapsed'].value() *
        kernel['sm__cycles_elapsed.avg.per_second'].value()
    )
    achieved_traffic = kernel['dram__bytes.sum.per_second'].value()
    return achieved_work, achieved_traffic

def peak(kernel):
    # The peak_sustained metric is actually hard-coded per GPU, so this should be
    # the actual theoretical peak performance for this instruction.
    peak_work = (
        kernel['sm__ops_path_tensor_src_fp16_dst_fp32_sparsity_off.sum.peak_sustained'].value() *
        kernel['device__attribute_max_gpu_frequency_khz'].value() * 1000
    )
    peak_bandwidth = (
        kernel['dram__bytes.sum.peak_sustained'].value() *
        kernel['dram__cycles_elapsed.avg.per_second'].value()
    )
    return peak_work, peak_bandwidth

def tflops(flops):
    return [a * 1e-12 for a in flops]

def plot_scatter(flops_per_second, flops_per_byte, label):
    plt.scatter(flops_per_second, flops_per_byte, lable=label)

def get_data(kernels, func):
    data = [func(kernel) for kernel in kernels]
    flops_per_second, bytes_per_second = zip(*data)
    flops_per_byte = [f / b for (f, b) in data]
    return flops_per_second, flops_per_byte, bytes_per_second

print([kernel.name() for kernel in context[0]])

scatter_kernels = [kernel for kernel in context[0] if 'scatter' in kernel.name()]
gemm_kernels = [kernel for kernel in context[0] if 'ampere' in kernel.name() or 'sm80' in kernel.name() or 'Kernel' in kernel.name()]

assert (len(scatter_kernels) == len(gemm_kernels))

ach_flops_per_second, ach_flops_per_byte, ach_bytes_per_second = get_data(scatter_kernels, acheived)
plt.scatter(ach_flops_per_byte, tflops(ach_flops_per_second), label='scatter2scatter Acheived', color='red', marker='x')

ach_flops_per_second, ach_flops_per_byte, _ = get_data(gemm_kernels, acheived)
plt.scatter(ach_flops_per_byte, tflops(ach_flops_per_second), label='torch.matmul Acheived', color='blue', marker='x')

def plot_roof(peak_flops_per_second, peak_bytes_per_second):
    arith_intensity = np.arange(1, 1000, device='cpu')
    memory_roof = arith_intensity * (peak_bytes_per_second[-1] * 1e-12)  # (FLOPS/Byte) / (Byte/second) == (FLOPS/second)
    compute_roof = np.full(arith_intensity.shape[0], 1e-12 * peak_flops_per_second[-1])
    plt.plot(arith_intensity, 
            [min(a, b) for (a, b) in zip(memory_roof, compute_roof)], 
            label='GPU Roofline', 
            color='gray', 
            linestyle='dashed')

peak_flops_per_second, peak_flops_per_byte, peak_bytes_per_second = get_data(gemm_kernels, peak)

plot_roof(peak_flops_per_second, peak_bytes_per_second)
plt.ylabel('TFLOPs/s')
plt.xlabel('Arith Intensity (FLOPs / byte)')
plt.xscale('log')
plt.title(torch.cuda.get_device_name())
plt.legend()
plt.savefig('roofline.png', dpi=300)