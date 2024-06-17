////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <type_traits>

namespace assera
{
namespace utilities
{
    /// <summary> A simple wrapper around bool to work around the std::vector<bool> specialization. </summary>
    struct Boolean
    {
        /// <summary> Default constructor. Constructs an instance with a wrapped false bool value. </summary>
        Boolean();

        /// <summary> Constructs an instance that wraps the provided bool value. </summary>
        /// <param name="value"> The bool value to wrap </param>
        Boolean(bool value);

        /// <summary> bool conversion operation </summary>
        operator bool() const { return value; }

    private:
        bool value = false;
    };
    static_assert(std::is_default_constructible_v<Boolean> &&
                  std::is_nothrow_move_assignable_v<Boolean> &&
                  std::is_nothrow_move_constructible_v<Boolean> &&
                  std::is_swappable_v<Boolean>);

    bool operator==(Boolean b1, Boolean b2);
    bool operator==(bool b1, Boolean b2);
    bool operator==(Boolean b1, bool b2);

    bool operator!=(Boolean b1, Boolean b2);
    bool operator!=(bool b1, Boolean b2);
    bool operator!=(Boolean b1, bool b2);

} // namespace utilities
} // namespace assera
