[//]: # (Project: Assera)
[//]: # (Version: v1.2)

# Assera v1.2 Reference

## `assera.Nest.iteration_logic(logic)`
Adds an iteration logic function to a `Nest`.

## Arguments

argument | description | type/default
--- | --- | ---
`logic` | | Python function that represents the logic to run in the innermost loop of the nest.

## Examples

The preferred syntax uses Python decorators, as follows:
```python
import assera as acc

A = acc.Array(role=acc.role.INPUT, shape=(16, 64))
B = acc.Array(role=acc.role.INPUT, shape=(64, 32))
C = acc.Array(role=acc.role.INPUT_OUTPUT, shape=(16, 32))

nest = acc.Nest(shape=(16, 32, 64))
i, j, k = nest.get_indices()

@nest.iteration_logic
def _():
    C[i,j] += A[i,k] * B[k,j]
```

The alternative syntax avoids decorators and instead defines the logic in a function:
```python
import assera as acc

A = acc.Array(role=acc.role.INPUT, shape=(16, 64))
B = acc.Array(role=acc.role.INPUT, shape=(64, 32))
C = acc.Array(role=acc.role.INPUT_OUTPUT, shape=(16, 32))

nest = acc.Nest(shape=(16, 32, 64))
i, j, k = nest.get_indices()

def logic_fn():
    C[i, j] += A[i, k] * B[k, j]

nest.iteration_logic(logic_fn)
```


<div style="page-break-after: always;"></div>
