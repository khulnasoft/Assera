////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "nest/AffineExpression.h"

namespace assera::ir
{
namespace loopnest
{
    bool AffineExpression::IsIdentity() const
    {
        return (_expr == nullptr);
    }
} // namespace loopnest
} // namespace assera::ir
