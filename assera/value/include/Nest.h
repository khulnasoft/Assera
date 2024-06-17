////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cassert>
#include <functional>
#include <memory>
 

#include "Index.h"
#include "IterationDomain.h"
#include "Range.h"
#include "Scalar.h"
#include "ScalarDimension.h"
#include "ScalarIndex.h"

#include <utilities/include/MemoryLayout.h>
#include <utilities/include/TupleUtils.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>

namespace assera::ir::loopnest
{
class NestOp;
}

namespace assera
{
namespace value
{
    using loopnests::Index;
    using loopnests::IterationDomain;
    using loopnests::Range;
    using utilities::MemoryLayout;

    using ScalarIndexPair = std::pair<ScalarIndex, ScalarIndex>;
    class Schedule;
    class NestImpl;
    class RuntimeNestImpl;

    class Nest
    {
    public:
        /// <summary> Constructor that creates a nest from a MemoryShape </summary>
        /// <param name="sizes"> Memory shape describing the sizes </param>
        /// <param name="runtimeSizes"> A vector of runtime sizes </param>
        Nest(const utilities::MemoryShape& sizes, const std::vector<ScalarDimension>& runtimeSizes = {});

        /// <summary> Constructor that creates a nest from a vector of ranges </summary>
        /// <param name="ranges"> A vector of assera::ir::loopnest::Range's </param>
        /// <param name="runtimeSizes"> A vector of runtime sizes </param>
        Nest(const std::vector<Range>& ranges, const std::vector<ScalarDimension>& runtimeSizes = {});

        Nest(const IterationDomain& domain, const std::vector<ScalarDimension>& runtimeSizes = {});

        Nest(Nest&& other);

        ~Nest();

        /// <summary> Returns the specified index for this nest, with the outermost index being index 0 </summary>
        ScalarIndex GetIndex(int pos);

        /// <summary> Returns the indices for this nest, starting from the outermost index </summary>
        std::vector<ScalarIndex> GetIndices();

        /// <summary> Returns the indices for this nest, starting from the outermost index </summary>
        template <int N>
        utilities::RepeatTuple<ScalarIndex, N> GetIndices();

        IterationDomain GetDomain() const;

        /// <summary> Sets the default kernel function to be run in the innermost loop </summary>
        void Set(std::function<void()> kernelFn);

        /// <summary> Creates a schedule to run this nest </summary>
        Schedule CreateSchedule();

        void dump();

    private:
        friend class Schedule;

        assera::ir::loopnest::NestOp GetOp();

        std::unique_ptr<NestImpl> _impl;
    };

} // namespace value
} // namespace assera

#pragma region implementation

namespace assera
{
namespace value
{
    template <int N>
    utilities::RepeatTuple<ScalarIndex, N> Nest::GetIndices()
    {
        using std::begin;
        using std::end;
        utilities::RepeatTuple<ScalarIndex, N> result;
        auto indices = GetIndices();
        assert(indices.size() >= N);

        return utilities::VectorToTuple<N>(indices);
    }

} // namespace value
} // namespace assera
