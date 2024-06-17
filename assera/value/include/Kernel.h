////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <array>
#include <cassert>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "Scalar.h"

#include <utilities/include/FunctionUtils.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>

namespace assera::ir::loopnest
{
class KernelOp;
}

namespace assera
{
namespace value
{
    class KernelImpl;

    class Kernel
    {
    public:
        Kernel( std::string id, std::function<void()> kernelFn);
        Kernel(Kernel&& other);
        ~Kernel();

        std::vector<Scalar> GetIndices() const;
        void dump();

    private:
        friend class Nest;
        friend class Schedule;
        assera::ir::loopnest::KernelOp GetOp() const;

        std::unique_ptr<KernelImpl> _impl;
    };

} // namespace value
} // namespace assera
#pragma region implementation
