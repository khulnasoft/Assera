#!/usr/bin/env python3
# Assera Tensor Alpha Scaling MatMul GPU sample: generator
import assera as acc
import tensor_matmul_gpu_generator
import matmul_utils as utils

mma_shape = acc.MMAShape.M16xN16xK4_B1
target = acc.Target(acc.Target.Model.AMD_MI100)
plan, A, B, C, tensor_indices = tensor_matmul_gpu_generator.create_basic_tensor_matmul_plan(target, mma_shape)

# Tensorize the plan
plan.tensorize(indices=tensor_indices, mma_shape=mma_shape, prologue_op=acc.MMAFragmentOp.SET, prologue_arg=0.0, epilogue_op=acc.MMAFragmentOp.SCALE, epilogue_arg=5.0)

utils.add_function_build_pkg(plan, A, B, C, "tensor_alpha_scaling_matmul_gpu")
