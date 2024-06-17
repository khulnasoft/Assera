////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors:  Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef ASSERA_accv_BASE
#define ASSERA_accv_BASE

include "ir/include/Common.td"

// Provide a definition of the 'value' dialect in the ODS framework so that we
// can define our operations.
def Value_Dialect : Dialect {
    let name = "accv";
    let cppNamespace = "assera::ir::value";
    let description = [{The Value (`accv`) dialect
    }];
    let extraClassDeclaration = [{
        static StringRef getAsseraMetadataAttrName() {
            return "accv.assera_metadata";
        }
    }];

    let hasConstantMaterializer = 1;
}

// Base class for value dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class accv_Op<string mnemonic, list<Trait> traits = []> : Op<Value_Dialect, mnemonic, traits>;

#endif // ASSERA_accv_BASE
