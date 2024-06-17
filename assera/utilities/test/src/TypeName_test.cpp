////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_all.hpp>

#include <utilities/include/TypeName.h>
#include <utilities/include/UniqueId.h>

#include <string>

namespace assera
{

enum SimpleEnum
{
    A,
    B,
    C
};

enum class ClassEnum
{
    a,
    b,
    c
};

TEST_CASE("TypeNames")
{
#define TEST_TYPE_NAMES(TYPE) \
    CHECK(utilities::TypeName<TYPE>::GetName() == #TYPE)

    TEST_TYPE_NAMES(bool);
    TEST_TYPE_NAMES(char);
    TEST_TYPE_NAMES(int);
    TEST_TYPE_NAMES(float);
    TEST_TYPE_NAMES(double);

    using utilities::UniqueId;
    TEST_TYPE_NAMES(UniqueId);

    CHECK(utilities::TypeName<SimpleEnum>::GetName() == "enum");
    CHECK(utilities::TypeName<ClassEnum>::GetName() == "enum");

#undef TEST_TYPE_NAMES
}

} // namespace assera
