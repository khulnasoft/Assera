[//]: # (Project: Assera)
[//]: # (Version: v1.2)

# Assera v1.2 Reference

## `assera.create_parameters()`
Creates placeholder parameters.

## Returns
Tuple of `Parameter`

## Examples

Create 3 parameters `m`, `n`, `k`. Use them to parameterize the nest shape:

```python
m, n, k = acc.create_parameters()
nest = acc.Nest(shape=(m, n, k))
```

<div style="page-break-after: always;"></div>


