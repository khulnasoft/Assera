////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef LOOPNEST_BASE
#define LOOPNEST_BASE

include "mlir/IR/OpBase.td"

// Provide a definition of the dialect in the ODS framework so that we can define our operations.
def LoopNest_Dialect : Dialect {
  let name = "accln";
  let description = [{The LoopNest (`accln`) dialect
  }];
  let cppNamespace = "assera::ir::loopnest";
  let hasConstantMaterializer = 1;
}

def rlcn_IsSymbolicIndexType :
  CPred<"$_self.isa<assera::ir::loopnest::SymbolicIndexType>()">;

def rlcn_SymbolicIndex : DialectType<LoopNest_Dialect,
                                        rlcn_IsSymbolicIndexType,
                                        "LoopNest symbolic index">;

def rlcn_IsArrayType :
  CPred<"$_self.isa<assera::ir::loopnest::ArrayType>()">;

def rlcn_LoopNestArrayType : DialectType<LoopNest_Dialect,
                                        rlcn_IsArrayType,
                                        "LoopNest Array Type">;

def rlcn_IsKernelType :
  CPred<"$_self.isa<assera::ir::loopnest::KernelType>()">;

def rlcn_KernelType : DialectType<LoopNest_Dialect,
                                        rlcn_IsKernelType,
                                        "LoopNest Kernel Type">;


#endif // LOOPNEST_BASE
