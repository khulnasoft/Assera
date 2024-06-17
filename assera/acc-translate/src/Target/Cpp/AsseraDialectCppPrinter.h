////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef ARGO_DIALECT_CPP_PRINTER_H_
#define ARGO_DIALECT_CPP_PRINTER_H_

// #include "CppPrinter.h"
// #include "mlir/Dialect/Argo/IR/ArgoOps.h"

#include <ir/include/argo/ArgoOps.h>
#include <ir/include/value/ValueDialect.h>

#include "CppPrinter.h"

namespace mlir
{
namespace cpp_printer
{

    struct AsseraDialectCppPrinter : public DialectCppPrinter
    {
        enum class MMAKernelKind
        {
            m8n8k4RowColfp32,
            InvalidKernel
        };

        AsseraDialectCppPrinter(CppPrinter* printer) :
            DialectCppPrinter(printer) {}

        std::string getName() override { return "Assera"; }

        LogicalResult printOp(assera::ir::value::MMAAllocSyncOp op);
        //LogicalResult printOp(assera::ir::value::MMAFillSyncOp op);
        LogicalResult printOp(assera::ir::value::MMALoadSyncOp op);
        LogicalResult printOp(assera::ir::value::MMAComputeSyncOp op);
        LogicalResult printOp(assera::ir::value::MMAStoreSyncOp op);
        LogicalResult printOp(assera::ir::value::GPUBlockCacheOp op);
        LogicalResult printOp(assera::ir::value::CallOp op);
        LogicalResult printOp(assera::ir::value::ReturnOp op);
        LogicalResult printOp(assera::ir::value::WarpIdOp warpIdOp);

        LogicalResult printDialectOperation(Operation* op, bool* skipped, bool* consumed) override;

        LogicalResult printIntrinsicCallOp(Operation* callOp, Operation* defFuncOp, bool* consumed) override;

        LogicalResult printPrologue() override;

        LogicalResult printEpilogue() override;

        LogicalResult runPrePrintingPasses(Operation* op) override;

        llvm::SmallVector<StringRef, 0> MMAKernelNames;

        llvm::SmallVector<FuncOp, 1> CudaKernels;

    private:
        LogicalResult printVectorType(mlir::Type elementType, const uint32_t stride) const;
    };

} // namespace cpp_printer
} // namespace mlir

#endif
