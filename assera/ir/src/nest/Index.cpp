////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Chuck Jacobs, Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "nest/Index.h"

namespace assera::ir
{
namespace loopnest
{
    Index Index::none = Index("", Index::DefaultID);

    Index::Index(const std::string& name) :
        Index(name, Index::GetNextId())
    {
    }

    Index::Index(const std::string& name, Id id) :
        _name(name),
        _id(id)
    {
    }

    const std::string& Index::GetName() const
    {
        return _name;
    }

    Index::Id Index::GetId() const
    {
        return _id;
    }

    // TODO: Change this so that IDs are the responsibility of the EmitterContext
    Index::Id Index::GetNextId()
    {
        static Id _nextIndex = 0;
        return _nextIndex++;
    }

    std::ostream& operator<<(std::ostream& os, const Index& index)
    {
        os << index.GetName() << "(" << index.GetId() << ")";
        return os;
    }

} // namespace loopnest
} // namespace assera::ir

using namespace assera::ir::loopnest;

std::hash<Index>::result_type std::hash<Index>::operator()(const argument_type& element) const
{
    return static_cast<size_t>(std::hash<Index::Id>()(element.GetId()));
}
