[//]: # (Project: Assera)
[//]: # (Version: v1.2)

# Assera v1.2 Reference

## `assera.Plan.tensorize(indices, mma_shape [, use_static_offsets, num_total_passes, num_fused_passes, scheduling_policy, prologue_op, epilogue_op])`
Only available for targets with native matrix multiplication instruction (tensor core) support. Marks the dimensions of the iteration-space for tensorization. Only perfectly nested loops of the following form can be tensorized:

```python
for i in range(M):
    for k in range(N):
        for j in range(K):
            C[i, j] += A[i, k] * B[k, j]
```

## Arguments

argument | description | type/default
--- | --- | ---
`indices` | The 3-dimensional iteration space to tensorize. | 3-D tuple of `assera.Index`
`mma_shape` | The type of MMA operation to use. | `assera.MMAShape`
`use_static_offsets` | This is an optimization flag, which when enabled will use precomputed offset maps stored in device constant memory. Defaults to `False`. | bool
`num_total_passes` | This controls the total number of passes to run. Defaults to 1. | positive integer
`num_fused_passes` | This controls the number of passes for which register allocation is done, higher the value more the number of registers that are allocated. Defaults to `None` which will fuse all the passes as specified by `num_total_passes`. | positive integer
`scheduling_policy` | For multi-block MMA operations, this controls whether matrix multiplication is done block-by-block or pass-by-pass (affects register usage). Default value is `assera.MMASchedulingPolicy.PASS_ORDER` | `assera.MMASchedulingPolicy`
`prologue_op` | The element-wise operation to apply on matrix fragment data as a part of initialization (pre-tensorization). Default value is `assera.MMAFragmentOp.NONE` | `assera.MMAFragmentOp`
`epilogue_op` | The element-wise operation to apply on matrix fragment data as a part of the final store (post-tensorization). Default value is `assera.MMAFragmentOp.NONE` | `assera.MMAFragmentOp`

The different values of the enum `MMAShape` are explained here: [`assera.MMAShape`](<../../enumerations/MMAShape.md>)

The different values of the enum `MMASchedulingPolicy` (applicable only for AMD targets supporting MFMA ops, such as `assera.Target.Model.AMD_MI100`) are mentioned here: [`assera.MMASchedulingPolicy`](<../../enumerations/MMASchedulingPolicy.md>)

The different values of the enum `MMAFragmentOp` are explained here: [`assera.MMAFragmentOp`](<../../enumerations/MMAFragmentOp.md>)

## Examples

Mark the dimensions `ii`, `jj`, and `kk` for tensorization execution:

```python
plan.tensorize(indices=(ii,jj,kk))
```

<div style="page-break-after: always;"></div>


