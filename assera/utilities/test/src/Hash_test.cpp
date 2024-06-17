////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_all.hpp>

#include <utilities/include/Hash.h>

#include <tuple>
#include <utility>

namespace assera
{

TEST_CASE("Hash")
{
    CHECK(std::hash<int>{}(3) == utilities::HashValue(3));

    size_t seed = 0;
    utilities::HashCombine(seed, 3);

    CHECK(seed == utilities::HashValue(std::tuple{ 3 }));

    REQUIRE(utilities::HashValue(std::vector{ 1, 2, 3 }) != utilities::HashValue(std::vector{ 3, 2, 1 }));
}

} // namespace assera
