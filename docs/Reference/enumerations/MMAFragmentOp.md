[//]: # (Project: Assera)
[//]: # (Version: v1.2)

# Assera v1.2 Reference
## `assera.MMAFragmentOp`

type | description | Mathematical formula
--- | --- | ---
`assera.MMAFragmentOp.NONE` | No-op which does not modify the fragment data. | `f(x) = x`
`assera.MMAFragmentOp.ReLU` | Rectified linear unit activation function ([details](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))). | `f(x) = max(0, x)`
`assera.MMAFragmentOp.ReLU_NoConditional` | Rectified linear unit activation function which does not generate divergent code. | `f(x) = x * bool(x > 0)`
`assera.MMAFragmentOp.SET` | Sets the data to scalar constant, C. | `f(x) = C`
`assera.MMAFragmentOp.SCALE` | Multiplies the data by a scalar constant, C. | `f(x) = C.f(x)`

<div style="page-break-after: always;"></div>
