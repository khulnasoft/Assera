[//]: # (Project: Assera)
[//]: # (Version: v1.2)

# Assera v1.2 Reference

## `assera.Array.Layout`

type | description
--- | ---
`assera.Array.Layout.FIRST_MAJOR` | Specifies a memory layout where the first major axis is in contiguous memory. For example, in a matrix, this corresponds to "row-major".
`assera.Array.Layout.LAST_MAJOR` | Specifies a memory layout where the last major axis is in contiguous memory. For example, in a matrix, this corresponds to "column-major".
`assera.Array.Layout.DEFERRED` | Defer specifying the memory layout for an `Role.CONST` array until a cache is created.

<div style="page-break-after: always;"></div>
