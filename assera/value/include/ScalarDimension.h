////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Scalar.h"

namespace assera
{
namespace value
{
    class ScalarDimension : public Scalar
    {
    public:
        ScalarDimension(Role role = Role::Input);
        ScalarDimension(const std::string& name, Role role = Role::Input);
        ScalarDimension(Value value, const std::string& name = "", Role role = Role::Input);
        ~ScalarDimension();

        virtual void SetValue(Value value) final;
    };
} // namespace value
} // namespace assera
