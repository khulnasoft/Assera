////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Index.h"

#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/Value.h>

#include <vector>

namespace assera::ir
{
namespace loopnest
{
    class AffineExpression
    {
    public:
        AffineExpression() = default;
        AffineExpression(mlir::AffineExpr expr, const std::vector<Index>& indices) :
            _expr(expr), _indices(indices) {}

        mlir::Value Apply(const std::vector<mlir::Value>& indexValue);

        bool IsIdentity() const;

        const std::vector<Index>& GetIndices() const { return _indices; }
        mlir::AffineExpr GetAffineExpr() const { return _expr; }

    private:
        mlir::AffineExpr _expr;
        std::vector<Index> _indices;
    };

    inline bool operator==(const AffineExpression& a, const AffineExpression& b)
    {
        return (a.GetIndices() == b.GetIndices()) && (a.GetAffineExpr() == b.GetAffineExpr());
    }

    inline bool operator!=(const AffineExpression& a, const AffineExpression& b)
    {
        return !(a == b);
    }

} // namespace loopnest
} // namespace assera::ir
