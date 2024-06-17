////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <mlir/IR/BuiltinOps.h>

#include <functional>
#include <string>

using SetupFunc = std::function<void()>;
using VerifyFunc = std::function<bool(mlir::OwningOpRef<mlir::ModuleOp>&, mlir::FuncOp&)>;

void RunTest(std::string testName, SetupFunc&& setupFunc, std::string verifyName, VerifyFunc&& verifyFunc);

//
// RUN_TEST macro
//
#define RUN_TEST(Test, Verify)                     \
    do                                             \
    {                                              \
        RunTest(#Test, (Test), #Verify, (Verify)); \
    } while (0)
