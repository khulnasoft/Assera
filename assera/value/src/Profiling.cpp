////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Profiling.h"
#include "EmitterContext.h"

namespace assera
{
namespace value
{
    void EnterProfileRegion(const std::string& regionName)
    {
        GetContext().EnterProfileRegion(regionName);
    }

    void ExitProfileRegion(const std::string& regionName)
    {
        GetContext().ExitProfileRegion(regionName);
    }

    void PrintProfileResults()
    {
        GetContext().PrintProfileResults();
    }

    ProfileRegion::ProfileRegion(const std::string& regionName) :
        _regionName(regionName)
    {
        EnterProfileRegion(_regionName);
    }

    ProfileRegion::~ProfileRegion()
    {
        ExitProfileRegion(_regionName);
    }
} // namespace value
} // namespace assera
