////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef ASSERA_OPS
#define ASSERA_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Assera_Dialect : Dialect {
  let name = "assera";
  let description = [{The Assera (`assera`) dialect
  }];
  let cppNamespace = "assera::ir::rc";
}

class RC_Op<string mnemonic, list<Trait> traits = []> :
    Op<Assera_Dialect, mnemonic, traits>;

def RC_GemmOp : RC_Op<"GEMM", [NoSideEffect]> {
  let summary = "Assera GEMM operation";
  let description = [{
  }];
  let arguments = (ins AnyTypeOf<[TensorOf<[F16,F32,F64,I32,I64]>, MemRefOf<[F16,F32,F64,I32,I64]>]>:$A,
    AnyTypeOf<[TensorOf<[F16,F32,F64,I32,I64]>, MemRefOf<[F16,F32,F64,I32,I64]>]>:$B,
    AnyTypeOf<[TensorOf<[F16,F32,F64,I32,I64]>, MemRefOf<[F16,F32,F64,I32,I64]>, NoneType]>:$C,
    DefaultValuedAttr<F32Attr, "1.0">:$alpha,
    DefaultValuedAttr<F32Attr, "1.0">:$beta,
    DefaultValuedAttr<I64Attr, "0">:$transA,
    DefaultValuedAttr<I64Attr, "0">:$transB);
  let results = (outs AnyTypeOf<[TensorOf<[F16,F32,F64,I32,I64]>, MemRefOf<[F16,F32,F64,I32,I64]>]>:$Y);
}

#endif // ASSERA_OPS
