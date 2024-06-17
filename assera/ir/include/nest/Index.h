////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors:  Chuck Jacobs, Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <ostream>
#include <string>
#include <vector>

namespace assera::ir
{
namespace loopnest
{
    /// <summary>
    /// A placeholder object representing a runtime variable used as the index for a loop (e.g., the 'i' in 'for(i = ...)').
    /// </summary>
    class Index
    {
    public:
        using Id = int;
        Index() = default;
        Index(const Index& other) = default;
        Index(Index&& other) = default;
        Index(const std::string& name);
        Index(const std::string& name, Id id);

        Index& operator=(const Index& other) = default;
        Index& operator=(Index&& other) = default;

        const std::string& GetName() const;
        Id GetId() const;

        static Index none;
        static constexpr Id DefaultID = -1;

    private:
        static int GetNextId();

        friend inline bool operator==(const Index& i1, const Index& i2) { return i1.GetId() == i2.GetId() && i1.GetName() == i2.GetName(); }
        friend inline bool operator!=(const Index& i1, const Index& i2) { return !(i1 == i2); }
        friend inline bool operator<(const Index& i1, const Index& i2) { return i1.GetId() < i2.GetId(); }

        std::string _name;
        Id _id = Index::DefaultID;
    };

    struct SplitIndex
    {
        Index outer;
        Index inner;

        friend inline bool operator==(const SplitIndex& i1, const SplitIndex& i2) { return i1.outer == i2.outer && i1.inner == i2.inner; }
    };

    std::ostream& operator<<(std::ostream& os, const Index& index);
} // namespace loopnest
} // namespace assera::ir

namespace std
{
template <>
struct hash<::assera::ir::loopnest::Index>
{
    using argument_type = ::assera::ir::loopnest::Index;
    using result_type = std::size_t;
    result_type operator()(const argument_type& index) const;
};
} // namespace std
