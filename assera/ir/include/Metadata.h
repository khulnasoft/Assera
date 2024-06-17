

#pragma once

#include <utilities/include/Boolean.h>

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>

#include <map>
#include <stdint.h>
#include <string>
#include <variant>
#include <vector>

namespace assera
{
namespace ir
{
    using MetadataValueType = std::variant<bool,
                                           int,
                                           int64_t,
                                           std::string,
                                           std::vector<utilities::Boolean>,
                                           std::vector<int>,
                                           std::vector<int64_t>,
                                           std::vector<std::string>>;

    using Metadata = std::map<std::string, MetadataValueType>;

    Metadata ParseFullMetadata(mlir::DictionaryAttr metadataDictionary);
    mlir::Attribute GetMetadataAttr(const MetadataValueType& value, mlir::MLIRContext* context);
} // namespace ir
} // namespace assera
