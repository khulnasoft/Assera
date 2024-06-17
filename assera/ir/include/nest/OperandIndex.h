

#pragma once

#include <ostream>
#include <string>
#include <vector>

namespace assera::ir
{
namespace loopnest
{
    /// <summary>
    /// An object referencing an associated operand of an op by index
    /// </summary>
    class OperandIndex
    {
    public:
        OperandIndex() = default;
        OperandIndex(const OperandIndex& other) = default;
        OperandIndex(OperandIndex&& other) = default;
        OperandIndex(int64_t index);

        OperandIndex& operator=(const OperandIndex& other) = default;
        OperandIndex& operator=(OperandIndex&& other) = default;

        int64_t GetIndex() const;

    private:
        friend inline bool operator==(const OperandIndex& i1, const OperandIndex& i2) { return i1.GetIndex() == i2.GetIndex(); }
        friend inline bool operator!=(const OperandIndex& i1, const OperandIndex& i2) { return !(i1 == i2); }
        friend inline bool operator<(const OperandIndex& i1, const OperandIndex& i2) { return i1.GetIndex() < i2.GetIndex(); }

        int64_t _idx = -1;
    };

    std::ostream& operator<<(std::ostream& os, const OperandIndex& index);
} // namespace loopnest
} // namespace assera::ir

namespace std
{
template <>
struct hash<::assera::ir::loopnest::OperandIndex>
{
    using argument_type = ::assera::ir::loopnest::OperandIndex;
    using result_type = std::size_t;
    result_type operator()(const argument_type& index) const;
};
} // namespace std
