////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Array.h"
#include "Scalar.h"

namespace assera
{
namespace value
{
    void CopyArray(Array A, Array B);
    void ClearArray(Array A);
    void FillArray(Array A, Scalar val);

    Scalar VectorMax(Array v);
    Scalar VectorSum(Array v);

    void ClearMatrix(Array A);
    void TransposeMatrix(Array A, Array B);

    void MatMulBasic(Array A, Array B, Array C, bool clearC = true);
    void MatMulSimpleTiled(Array A, Array B, Array C, bool clearC = true);
    void MatMulMlas(Array A, Array B, Array C, bool clearC = true);
} // namespace value
} // namespace assera
