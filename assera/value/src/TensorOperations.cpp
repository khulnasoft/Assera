////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "TensorOperations.h"
#include "EmitterContext.h"
#include "Scalar.h"
#include "Tensor.h"

namespace assera
{
using namespace utilities;

namespace value
{
    Scalar Sum(Tensor tensor)
    {
        Scalar result = Allocate(tensor.GetType(), ScalarLayout);

        For(tensor, [&](auto row, auto column, auto channel) {
            result += tensor(row, column, channel);
        });

        return result;
    }

    void For(Tensor tensor, std::function<void(Scalar, Scalar, Scalar)> fn)
    {
        For(std::string{}, tensor, fn);
    }

    void For(const std::string& name, Tensor tensor, std::function<void(Scalar, Scalar, Scalar)> fn)
    {
        auto layout = tensor.GetValue().GetLayout();
        if (layout.NumDimensions() != 3)
        {
            throw InputException(InputExceptionErrors::invalidArgument,
                                 "Layout being looped over must be three-dimensional");
        }

        GetContext().For(
            layout,
            [fn = std::move(fn)](std::vector<Scalar> coordinates) {
                fn(coordinates[0], coordinates[1], coordinates[2]);
            },
            name);
    }

} // namespace value
} // namespace assera
