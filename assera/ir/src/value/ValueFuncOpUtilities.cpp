////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////

#include "value/ValueDialect.h"

#include "IRUtil.h"

namespace assera::ir::value
{

ValueFuncOp CreateRawPointerAPIWrapperFunction(mlir::OpBuilder& builder, ValueFuncOp functionToWrap, mlir::StringRef wrapperFnName)
{
    auto loc = functionToWrap.getLoc();
    mlir::OpBuilder::InsertionGuard insertGuard(builder);

    ValueModuleOp vModuleOp = functionToWrap->getParentOfType<ValueModuleOp>();

    auto insertionPoint = assera::ir::util::GetTerminalInsertPoint<ValueModuleOp, ModuleTerminatorOp>(vModuleOp);
    builder.restoreInsertionPoint(insertionPoint);

    ValueFuncOp apiWrapperFn = builder.create<ValueFuncOp>(loc, wrapperFnName, functionToWrap.getType(), ir::value::ExecutionTarget::CPU );
    apiWrapperFn->setAttr(ir::HeaderDeclAttrName, builder.getUnitAttr());
    apiWrapperFn->setAttr(ir::RawPointerAPIAttrName, builder.getUnitAttr());

    builder.setInsertionPointToStart(&apiWrapperFn.body().front());

    auto launchFuncOp = builder.create<LaunchFuncOp>(loc, functionToWrap, apiWrapperFn.getArguments());

    if (launchFuncOp.getNumResults() > 0)
    {
        builder.create<ReturnOp>(loc, launchFuncOp.getResults() );
    }
    else
    {
        builder.create<ReturnOp>(loc);
    }
    return apiWrapperFn;
}

} // namespace assera::ir::value
