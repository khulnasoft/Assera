[//]: # (Project: Assera)
[//]: # (Version: v1.2)

# Assera v1.2 Reference

## `assera.Dimension([role, value])`
Constructs a runtime dimension size with optional initialization.

Note: This constructor is meant for advanced use cases that involve Python generator expressions. For the simplified syntax to create dimensions, see [create_dimensions](../../functions/create_dimensions.md).

## Arguments

argument | description | type/default
--- | --- | ---
`role` | The role of the dimension determines if it is mutable or immutable. | [`assera.Role`](<../../enumerations/Role.md>). default: `assera.Role.INPUT`.
`name` | The name of the dimension variable. Default is an empty string. | string
`value` | The optional value to initialize the dimension. Only applies to mutable dimensions (`assera.Role.OUTPUT`) | integer or `Dimension`

## Returns
`Dimension`

## Examples

Construct an output array with runtime dimensions using Python tuple comprehension over an input shape:
```python
import assera as acc

# input_shape is a tuple or list of acc.Dimensions or integers
output_shape = tuple(acc.Dimension(role=acc.Role.OUTPUT, value=i) for i in input_shape)
A = acc.Array(role=acc.Role.OUTPUT, element_type=acc.ScalarType.float32, shape=output_shape)
```

<div style="page-break-after: always;"></div>