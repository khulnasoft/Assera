////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <functional>

namespace mlir
{
class FuncOp;
class OpBuilder;

template <typename OpTy>
class OwningOpRef;
class ModuleOp;
}

void SetTestBuilder(mlir::OpBuilder* builder);
mlir::OpBuilder& GetTestBuilder();

//
// Test function verifiers
//
bool VerifyGenerate(mlir::OwningOpRef<mlir::ModuleOp>& module, mlir::FuncOp& fn, std::string outputFile="");
bool VerifyParse(mlir::OwningOpRef<mlir::ModuleOp>& module, mlir::FuncOp& fn, std::string outputFile="");
bool VerifyLowerToStd(mlir::OwningOpRef<mlir::ModuleOp>& module, mlir::FuncOp& fn, std::string outputFile="");
bool VerifyLowerToLLVM(mlir::OwningOpRef<mlir::ModuleOp>& module, mlir::FuncOp& fnOp, std::string outputFile="");
bool VerifyTranslateToLLVMIR(mlir::OwningOpRef<mlir::ModuleOp>& module, mlir::FuncOp& fnOp, bool optimize, std::string outputFile="");

