////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Umesh Madan, Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace assera
{
namespace value
{
    ///<summary> An enumeration of primitive types our compilers support </summary>
    enum class VariableType
    {
        Void = 0,
        ///<summary> 1 bit boolean </summary>
        Boolean,
        ///<summary> 8 bit unsigned integer </summary>
        Byte,
        ///<summary> 8 bit signed integer </summary>
        Int8,
        ///<summary> 16 bit signed integer </summary>
        Int16,
        ///<summary> 32 bit signed integer </summary>
        Int32,
        ///<summary> 64 bit signed integer </summary>
        Int64,
        ///<summary> 4 byte floating point </summary>
        Float,
        ///<summary> 8 byte floating point </summary>
        Double,

        //
        // Pointers
        //
        VoidPointer,
        ///<summary> Pointer to a byte </summary>
        BytePointer,
        ///<summary> Pointer to an 8 bit signed integer </summary>
        Int8Pointer,
        ///<summary> Pointer to a Int16 </summary>
        Int16Pointer,
        ///<summary> Pointer to an Int32 </summary>
        Int32Pointer,
        ///<summary> Pointer to an Int64 </summary>
        Int64Pointer,
        ///<summary> Pointer to a Float </summary>
        FloatPointer,
        ///<summary> Pointer to a Double </summary>
        DoublePointer,

        //
        // Pointers to pointers
        //
        VoidPointerPointer,
        ///<summary> Pointer to a pointer to a byte </summary>
        BytePointerPointer,
        ///<summary> Pointer to a pointer to an 8 bit signed integer </summary>
        Int8PointerPointer,
        ///<summary> Pointer to a pointer to a Int16 </summary>
        Int16PointerPointer,
        ///<summary> Pointer to a pointer to an Int32 </summary>
        Int32PointerPointer,
        ///<summary> Pointer to a pointer to an Int64 </summary>
        Int64PointerPointer,
        ///<summary> Pointer to a pointer to a Float </summary>
        FloatPointerPointer,
        ///<summary> Pointer to a pointer to a Double </summary>
        DoublePointerPointer,

        //
        // Custom Structs
        //
        Custom
    };

    /// <summary> Untyped unary operations. </summary>
    enum class UnaryOperatorType
    {
        none,
        logicalNot, // bool only
    };

    /// <summary> Untyped binary operations. </summary>
    enum class BinaryOperatorType
    {
        none,
        add,
        subtract,
        multiply,
        divide,
        logicalAnd,
        logicalOr,
        logicalXor,
        shiftLeft,
        logicalShiftRight,
        arithmeticShiftRight
    };

    /// <summary> Untyped binary predicates. </summary>
    enum class BinaryPredicateType
    {
        none,
        equal,
        less,
        greater,
        notEqual,
        lessOrEqual,
        greaterOrEqual
    };

    /// <summary> Untyped ternary operations. </summary>
    enum class TernaryOperationType
    {
        none,
        fma // fused multiply-add
    };

    ///<summary> An enumeration of strongly-typed operations on numbers </summary>
    enum class TypedOperator
    {
        none = 0,
        ///<summary> Integer addition </summary>
        add,
        ///<summary> Integer subtraction </summary>
        subtract,
        ///<summary> Integer multiplication </summary>
        multiply,
        ///<summary> Integer signed division - returns an integer </summary>
        divideSigned,
        ///<summary> modulo </summary>
        moduloSigned,
        ///<summary> Floating point addition </summary>
        addFloat,
        ///<summary> Floating point subtraction </summary>
        subtractFloat,
        ///<summary> Floating point multiplication </summary>
        multiplyFloat,
        ///<summary> Floating point division </summary>
        divideFloat,
        ///<summary> Binary and </summary>
        logicalAnd,
        ///<summary> Binary or </summary>
        logicalOr,
        ///<summary> Xor </summary>
        logicalXor,
        ///<summary> Bit-shift left </summary>
        shiftLeft,
        ///<summary> Bit-shift right, padding with zeros </summary>
        logicalShiftRight,
        ///<summary> Bit-shift right, extending sign bit </summary>
        arithmeticShiftRight
    };

    ///<summary> An enumeration of strongly TYPED comparisons on numbers </summary>
    enum class TypedComparison
    {
        none = 0,
        ///<summary> Integer Equal </summary>
        equals,
        ///<summary> Integer Less than </summary>
        lessThan,
        ///<summary> Integer Less than equals </summary>
        lessThanOrEquals,
        ///<summary> Integer Greater than </summary>
        greaterThan,
        ///<summary> Integer Greater than equals </summary>
        greaterThanOrEquals,
        ///<summary> Integer Not Equals </summary>
        notEquals,
        ///<summary> Floating point Equal </summary>
        equalsFloat,
        ///<summary> Floating point less than  </summary>
        lessThanFloat,
        ///<summary> Floating point less than equals </summary>
        lessThanOrEqualsFloat,
        ///<summary> Floating point greater than </summary>
        greaterThanFloat,
        ///<summary> Floating point less than equals </summary>
        greaterThanOrEqualsFloat,
        ///<summary> Floating point Not equals </summary>
        notEqualsFloat
    };

    /// <summary> Translate the unary operation operator into a strongly typed operator for LLVM </summary>
    ///
    /// <typeparam name="T"> The type of the value to operate on. </param>
    /// <param name="operation"> The (untyped) unary operation. </param>
    ///
    /// <returns> The typed version of the operation for the given type. </param>
    template <typename T>
    TypedOperator GetOperator(value::UnaryOperatorType operation);

    /// <summary> Translate the binary operation operator into a strongly typed operator for LLVM </summary>
    ///
    /// <typeparam name="T"> The type of the values to operate on. </param>
    /// <param name="operation"> The (untyped) binary operation. </param>
    ///
    /// <returns> The typed version of the operation for the given type. </param>
    template <typename T>
    TypedOperator GetOperator(value::BinaryOperatorType operation);

    /// <summary> Translate the binary operation operator into a strongly typed operator for LLVM </summary>
    ///
    /// <param name="operation"> The (untyped) binary operation. </param>
    ///
    /// <returns> The typed version of the operation for floating-point types. </param>
    TypedOperator GetFloatOperator(value::BinaryOperatorType operation);

    /// <summary> Translate the binary operation operator into a strongly typed operator for LLVM </summary>
    ///
    /// <param name="operation"> The (untyped) binary operation. </param>
    ///
    /// <returns> The typed version of the operation for integral types. </param>
    TypedOperator GetIntegerOperator(value::BinaryOperatorType operation);

    /// <summary> Translate the boolean binary operation operator into a strongly typed operator for LLVM </summary>
    ///
    /// <param name="operation"> The (untyped) binary operation. </param>
    ///
    /// <returns> The typed version of the boolean operation for boolean types. </param>
    TypedOperator GetBooleanOperator(value::BinaryOperatorType operation);

    /// <summary> Translate the binary predicate operator into a more strongly typed operator for LLVM </summary>
    ///
    /// <typeparam name="T"> The type of the values to compare. </param>
    /// <param name="predicate"> The (untyped) binary comparison. </param>
    ///
    /// <returns> The typed version of the comparison for the given type. </param>
    template <typename T>
    TypedComparison GetComparison(value::BinaryPredicateType predicate);

    /// <summary> Translate the binary predicate operator into a more strongly typed operator for LLVM </summary>
    ///
    /// <param name="predicate"> The (untyped) binary comparison. </param>
    ///
    /// <returns> The typed version of the comparison for floating-point types. </param>
    TypedComparison GetFloatComparison(value::BinaryPredicateType predicate);

    /// <summary> Translate the binary predicate operator into a more strongly typed operator for LLVM </summary>
    ///
    /// <param name="predicate"> The (untyped) binary comparison. </param>
    ///
    /// <returns> The typed version of the comparison for integral types. </param>
    TypedComparison GetIntegerComparison(value::BinaryPredicateType predicate);

    ///<summary> Commonly used to create named fields, arguments, variables </summary>
    using NamedVariableType = std::pair<std::string, VariableType>;

    ///<summary> Collections of variable types </summary>
    using VariableTypeList = std::vector<VariableType>;

    ///<summary> Collections of named variable types </summary>
    using NamedVariableTypeList = std::vector<NamedVariableType>;

    /// <summary> Gets the VariableType enum that corresponds to a given native C++ type. </summary>
    ///
    /// <typeparam name="ValueType"> The native C++ type being mapped to a value of VariableType. </typeparam>
    ///
    /// <returns> A VariableType that corresponds to a given native C++ type. </returns>
    template <typename ValueType>
    VariableType GetVariableType();

    /// <summary> Gets the value from the VariableType enum that corresponds to a pointer from a given nonpointer type. </summary>
    ///
    /// <param name="type"> The nonpointer type, such as Int16 or Double. </typeparam>
    ///
    /// <returns> A VariableType that corresponds to the pointer to a given type. </returns>
    VariableType GetPointerType(VariableType type);

    /// <summary> Gets the value from the VariableType enum that corresponds to a nonpointer from a given pointer type. </summary>
    ///
    /// <param name="type"> The pointer type, such as Int16Pointer or DoublePointer. </typeparam>
    ///
    /// <returns> A VariableType that corresponds to the nonpointer from a given type. </returns>
    VariableType GetNonPointerType(VariableType type);

    /// <summary> Gets the default value for a certain type. </summary>
    ///
    /// <typeparam name="ValueType"> The type. </typeparam>
    ///
    /// <returns> The default value for the given type. </returns>
    template <typename ValueType>
    ValueType GetDefaultValue();

    /// <summary> Gets the type-specific add element of the TypedOperator enum. </summary>
    ///
    /// <typeparam name="ValueType"> The type. </typeparam>
    ///
    /// <returns> The add operation that corresponds to the given type. </returns>
    template <typename ValueType>
    TypedOperator GetAddForValueType();

    /// <summary> Gets the type-specific subtract element of the TypedOperator enum. </summary>
    ///
    /// <typeparam name="ValueType"> The type. </typeparam>
    ///
    /// <returns> The subtract operation that corresponds to the given type. </returns>
    template <typename ValueType>
    TypedOperator GetSubtractForValueType();

    /// <summary> Gets the type-specific multiply element of the TypedOperator enum. </summary>
    ///
    /// <typeparam name="ValueType"> The type. </typeparam>
    ///
    /// <returns> The multiply operation that corresponds to the given type. </returns>
    template <typename ValueType>
    TypedOperator GetMultiplyForValueType();

    /// <summary> Gets the type-specific divide element of the TypedOperator enum. </summary>
    ///
    /// <typeparam name="ValueType"> The type. </typeparam>
    ///
    /// <returns> The divide operation that corresponds to the given type. </returns>
    template <typename ValueType>
    TypedOperator GetDivideForValueType();

    /// <summary> Gets the type-specific modulo element of the TypedOperator enum. </summary>
    ///
    /// <typeparam name="ValueType"> The type. </typeparam>
    ///
    /// <returns> The modulo operation that corresponds to the given type. </returns>
    template <typename ValueType>
    TypedOperator GetModForValueType();

    /// <summary> Does the given primitive type have a sign? </summary>
    ///
    /// <param name="type"> The type. </param>
    ///
    /// <returns> true if signed, false if not. </returns>
    bool IsSigned(VariableType type);

    /// <summary> Helper struct for getting the backing value type for a variable </summary>
    template <typename T>
    struct VariableValueType
    {
        using type = T;
        using DestType = type;

        static std::vector<DestType> ToVariableVector(const std::vector<T>& src);
        static std::vector<T> FromVariableVector(const std::vector<DestType>& src);
    };

    template <>
    struct VariableValueType<bool>
    {
        using type = int;
        using DestType = type;

        static std::vector<DestType> ToVariableVector(const std::vector<bool>& src);
        static std::vector<bool> FromVariableVector(const std::vector<DestType>& src);
    };
} // namespace value
} // namespace assera

#pragma region implementation

namespace assera
{
namespace value
{
    template <typename T>
    std::vector<typename VariableValueType<T>::DestType> VariableValueType<T>::ToVariableVector(const std::vector<T>& src)
    {
        return src;
    }

    template <typename T>
    std::vector<T> VariableValueType<T>::FromVariableVector(const std::vector<typename VariableValueType<T>::DestType>& src)
    {
        return src;
    }

    // bool specialization
    inline std::vector<typename VariableValueType<bool>::DestType> VariableValueType<bool>::ToVariableVector(const std::vector<bool>& src)
    {
        std::vector<VariableValueType<bool>::DestType> result(src.begin(), src.end());
        return result;
    }

    inline std::vector<bool> VariableValueType<bool>::FromVariableVector(const std::vector<typename VariableValueType<bool>::DestType>& src)
    {
        std::vector<bool> result(src.begin(), src.end());
        return result;
    }
} // namespace value
} // namespace assera

#pragma endregion implementation
