[//]: # (Project: Assera)
[//]: # (Version: v1.2)

# Assera v1.2 Reference

## `assera.MMASchedulingPolicy`

type | description
--- | ---
`assera.MMASchedulingPolicy.PASS_ORDER` | Process pass groups ([fused passes](../../Tutorials/GPU/Tensor_MatMul_GPU.md#tuning-parameters)) sequentially, within each pass group compute all the MFMA blocks. This allocates Accmulator registers required for all the blocks, however it only allocates input (A, B) registers which are only required for the current pass group.
`assera.MMASchedulingPolicy.BLOCK_ORDER` | Process MFMA blocks sequentially, for each block iterate over all the passes. This allocates Accumulator registers required for only 1 block and input (A, B) registers required for the entire pass group currently being processed. In this mode, input data for the same pass group is loaded into registers multiple times, once per block.

<div style="page-break-after: always;"></div>
