////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace assera
{
namespace value
{
    void EnterProfileRegion(const std::string& regionName);
    void ExitProfileRegion(const std::string& regionName);
    void PrintProfileResults();

    class ProfileRegion
    {
    public:
        explicit ProfileRegion(const std::string& regionName);
        ~ProfileRegion();

    private:
        std::string _regionName;
    };
} // namespace value
} // namespace assera
