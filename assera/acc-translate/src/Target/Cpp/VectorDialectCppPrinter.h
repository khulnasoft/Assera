////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Abdul Dakkak
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef VECTOR_DIALECT_CPP_PRINTER_H_
#define VECTOR_DIALECT_CPP_PRINTER_H_

#include "CppPrinter.h"

#include <mlir/Dialect/Vector/IR/VectorOps.h>

namespace mlir
{
namespace cpp_printer
{

    struct VectorDialectCppPrinter : public DialectCppPrinter
    {
        VectorDialectCppPrinter(CppPrinter* printer_) :
            DialectCppPrinter(printer_) {}

        std::string getName() override { return "Vector"; }

        /// print Operation from GPU Dialect
        LogicalResult printDialectOperation(Operation* op, bool* skipped, bool* consumed) override;
        LogicalResult printExtractElementOp(vector::ExtractElementOp op);
        LogicalResult printInsertElementOp(vector::InsertElementOp op);
        LogicalResult printLoadOp(vector::LoadOp op);
        LogicalResult printStoreOp(vector::StoreOp op);
        LogicalResult printBroadcastOp(vector::BroadcastOp op);
    };

} // namespace cpp_printer
} // namespace mlir

#endif
