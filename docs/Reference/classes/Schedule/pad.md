[//]: # (Project: Assera)
[//]: # (Version: v1.2)

# Assera v1.2 Reference

## `assera.Schedule.pad(index, size)`
Pads the beginning of a specified dimension of the iteration-space with empty (no-op) elements.

## Arguments

argument | description | type/default
--- | --- | ---
`index` | The dimension to pad | `Index`
`size` | The number of elements to pad | non-negative integer

## Examples

Pads the beginning of dimension `i` with 10 empty elements

```python
schedule.pad(i, 10)
```

<div style="page-break-after: always;"></div>
