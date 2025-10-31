# Based on intra-kernel profiling tutorual:
# https://github.com/triton-lang/triton/commit/48ff763d19e3efa41d94da31614c4b50afa51e09.
# Basically get the ttgir for a kernel, add some scopes, force triton to run the instrumented ttgir.
# I think this "manual" instrumentation is necessary because pl.scope doesn't work inside triton for loops?

import os
import pathlib
import torch
from triton.knobs import cache_knobs, compilation_knobs
import triton.profiler as proton
from moe_explore.testing import perfect_routing
from moe_explore.expert_permute import get_token_indices
from moe_explore.triton_kernels.m_grouped_gemm import m_grouped_gemm, MGroupedGEMMParams
from moe_explore.triton_kernels.autotune_config import AutotuneMode

def main():
    # setup inputs before instrumenting, in case pytorch ops happen to use triton
    input, weight, group_indices, params = setup_inputs(num_tokens=2048)
    # have triton dump its compiled code
    dumpdir = pathlib.Path("ttgir_dump/")
    dumpdir.mkdir(parents=True, exist_ok=True)
    
    compilation_knobs.always_compile = True
    compilation_knobs.dump_ir = True
    cache_knobs.dump_dir = str(dumpdir)
    m_grouped_gemm(input, weight, group_indices, params, AutotuneMode.NONE)

    # Iterate over all subdirectories in dumpdir and remove all flies except the .ttgir files
    for sub in dumpdir.rglob("*"):
        if not sub.is_dir() and sub.suffix != ".ttgir":
            sub.unlink()        
    print(f"TTGIR files dumped to {dumpdir}")
    
    # insert proton records into the dumped ttgir.
    for file in dumpdir.rglob("*.ttgir"):
        add_proton_records(file)
    print("TGGIR files instrumented")

    # re-run the code, but have triton use the instrumented ttgir
    compilation_knobs.always_compile = True
    compilation_knobs.dump_ir = False
    compilation_knobs.override = True
    cache_knobs.override_dir = dumpdir
    proton.start("m_grouped_gemm", data="trace", backend="instrumentation")
    m_grouped_gemm(input, weight, group_indices, params, AutotuneMode.NONE)
    proton.finalize()

def add_proton_records(file):
    # This is based on the instrumentation tutorial.
    with open(file, "r") as f:
        content = f.read()
        lines = f.readlines()
        
    if "proton.record" in content:
        raise AssertionError("file already has proton records")

    # reset file pointers
    with open(file, "r") as f:
        lines = f.readlines()
    
    loop_counter = 0    
    
    result_lines = []
    load_and_add_start = False
    for i, line in enumerate(lines):
        # Add kernel record start after function declaration
        if "tt.func public @" in line and "{" in line:
            result_lines.append(line)
            result_lines.append('   proton.record start "kernel"\n')
            continue
        
        # Add kernel record end before return
        if "tt.return" in line:
            result_lines.append('   proton.record end "kernel"\n')
            result_lines.append(line)
            continue
        
        # ttgir seems to have scf.for { ... yield },
        # so this adds a record after the for, and before the yield.
        if "scf.for" in line:
            result_lines.append(line)
            result_lines.append(f'   proton.record start "loop{loop_counter}"\n')
            loop_counter += 1
            continue
        if "scf.yield" in line:
            result_lines.append(f'  proton.record end "loop{loop_counter - 1}"\n')
            result_lines.append(line)
            loop_counter -= 1
            continue
        
        # Default: just add the line
        result_lines.append(line)
    
    with open(file, "w") as f:
        f.writelines(result_lines)

    print(f"Proton records added to {file}")

def setup_inputs(num_tokens):
    hidden_dim = 2048
    intermediate_dim = 768
    num_experts = 128
    topk = 8

    topk_scores, topk_indices = perfect_routing(num_tokens, num_experts, topk, device="cuda", dtype=torch.bfloat16)
    p = get_token_indices(
        topk_indices.view(-1),
        topk,
        num_experts,
        zero_prefix=True
    )

    num_tokens = num_tokens * topk
    input = torch.randn((num_tokens, hidden_dim), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((num_experts, hidden_dim, intermediate_dim), device="cuda", dtype=torch.bfloat16)
    params = MGroupedGEMMParams(
        permute_indices=None,
        gather=False,
        scatter=False,
        num_tokens=num_tokens,
        topk=topk,
        scales=None,
        activation=None
    )
    
    return input, weight, p.group_indices, params

if __name__ == "__main__":
    main()