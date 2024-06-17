

#include "AsseraPasses.h"

#include <value/include/FunctionDeclaration.h>

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>

#include <llvm/ADT/SmallString.h>
#include <llvm/Support/FormatVariadic.h>

#include <string>

using namespace mlir;

namespace
{

class FunctionPointerResolutionPass
    : public assera::transforms::FunctionPointerResolutionBase<FunctionPointerResolutionPass>
{
public:
    FunctionPointerResolutionPass() {}

    void runOnModule() override;

private:
    StringAttr GetFuncSymbolName(LLVM::LLVMFuncOp& op)
    {
        auto symbolNameAttr = op->getAttrOfType<StringAttr>(mlir::SymbolTable::getSymbolAttrName());
        return symbolNameAttr;
    }

    bool HasAsseraTemporaryPrefix(LLVM::LLVMFuncOp& op)
    {
        StringAttr funcSymbolName = GetFuncSymbolName(op);
        std::string strSymbolName = funcSymbolName.getValue().str();
        return strSymbolName.find(assera::value::FunctionDeclaration::GetTemporaryFunctionPointerPrefix(), 0) == 0;
    }

    std::string GetSymbolNameWithoutAsseraTemporaryPrefix(LLVM::LLVMFuncOp& op)
    {
        std::string strSymbolName = GetFuncSymbolName(op).getValue().str();
        assert(strSymbolName.find(assera::value::FunctionDeclaration::GetTemporaryFunctionPointerPrefix(), 0) == 0);
        return strSymbolName.substr(assera::value::FunctionDeclaration::GetTemporaryFunctionPointerPrefix().length());
    }
};

} // namespace

void FunctionPointerResolutionPass::runOnModule()
{
    // Find and replace usages of LLVM::LLVMFuncOp's prefixed with the temporary function pointer prefix with their non-prefixed counterparts
    SymbolTable symbolTable = SymbolTable::getNearestSymbolTable(getOperation());
    getOperation().walk([&](LLVM::LLVMFuncOp op) {
        if (HasAsseraTemporaryPrefix(op))
        {
            std::string replacementFuncName = GetSymbolNameWithoutAsseraTemporaryPrefix(op);
            auto replacementFunc = symbolTable.lookup(replacementFuncName);
            auto replacementLLVMFunc = dyn_cast<LLVM::LLVMFuncOp>(replacementFunc);

            [[maybe_unused]] auto ignored = symbolTable.replaceAllSymbolUses(op, GetFuncSymbolName(replacementLLVMFunc), getOperation());
        }
    });
}

namespace assera::transforms::value
{
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createFunctionPointerResolutionPass()
{
    return std::make_unique<FunctionPointerResolutionPass>();
}
} // namespace assera::transforms::value
