

#include "nest/OperandIndex.h"

namespace assera::ir
{
namespace loopnest
{
    OperandIndex::OperandIndex(int64_t index) :
        _idx(index)
    {
    }

    int64_t OperandIndex::GetIndex() const
    {
        return _idx;
    }

    std::ostream& operator<<(std::ostream& os, const OperandIndex& index)
    {
        os << index.GetIndex();
        return os;
    }

} // namespace loopnest
} // namespace assera::ir

using namespace assera::ir::loopnest;

std::hash<OperandIndex>::result_type std::hash<OperandIndex>::operator()(const argument_type& element) const
{
    return static_cast<size_t>(std::hash<int64_t>()(element.GetIndex()));
}
