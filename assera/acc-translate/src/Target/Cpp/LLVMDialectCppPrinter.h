////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Abdul Dakkak
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef LLVM_DIALECT_CPP_PRINTER_H_
#define LLVM_DIALECT_CPP_PRINTER_H_

#include "CppPrinter.h"

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

namespace mlir
{
namespace cpp_printer
{

    struct LLVMDialectCppPrinter : public DialectCppPrinter
    {
        LLVMDialectCppPrinter(CppPrinter* printer_) :
            DialectCppPrinter(printer_) {}

        std::string getName() override { return "LLVM"; }

        /// print Operation from GPU Dialect
        LogicalResult printDialectOperation(Operation* op, bool* skipped, bool* consumed) override;
        LogicalResult printFenceOp(LLVM::FenceOp op);
    };

} // namespace cpp_printer
} // namespace mlir

#endif
