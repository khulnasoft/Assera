////////////////////////////////////////////////////////////////////////////////////////////////////


//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "EmitterContext.h"
#include "Scalar.h"
#include "Value.h"
#include "VectorOperations.h"

#include <utilities/include/MemoryLayout.h>

#include <functional>

namespace assera
{
namespace value
{

    /// <summary> A View type that wraps a Value instance and enforces a memory layout that represents a vector </summary>
    class Vector
    {
    public:
        Vector();

        /// <summary> Constructor that wraps the provided instance of Value </summary>
        /// <param name="value"> The Value instance to wrap </param>
        /// <param name="name"> The optional name </param>
        Vector(Value value, const std::string& name = "");

        /// <summary> Constructs an instance from a vector of fundamental types </summary>
        /// <typeparam name="T"> Any fundamental type accepted by Value </typeparam>
        /// <param name="data"> The data to wrap </param>
        /// <param name="name"> The optional name </param>
        template <typename T, typename A = std::allocator<T>>
        Vector(std::vector<T, A> data, const std::string& name = "") :
            Vector(Value(std::move(data)), name)
        {}

        /// <summary> Constructs an instance from an initializer list of fundamental types </summary>
        /// <typeparam name="T"> Any fundamental type accepted by Value </typeparam>
        /// <param name="data"> The data to wrap </param>
        /// <param name="name"> The optional name </param>
        template <typename T>
        Vector(std::initializer_list<T> data, const std::string& name = "") :
            Vector(std::vector<T>(data), name)
        {}

        Vector(const Vector&);
        Vector(Vector&&) noexcept;
        Vector& operator=(const Vector&);
        Vector& operator=(Vector&&);
        ~Vector();

        /// <summary> Returns a Scalar value that represents the data at the specified index within the vector </summary>
        /// <param name="index"> The value by which to offset into the vector and return the specified value </param>
        /// <returns> The Scalar value wrapping the value that is at the specified index within the vector </returns>
        Scalar operator[](Scalar index);

        /// <summary> Returns a Scalar value that represents the data at the specified index within the vector </summary>
        /// <param name="index"> The value by which to offset into the vector and return the specified value </param>
        /// <returns> The Scalar value wrapping the value that is at the specified index within the vector </returns>
        Scalar operator()(Scalar index);

        /// <summary> Gets the underlying wrapped Value instance </summary>
        Value GetValue() const;

        /// <summary> Returns a subvector that starts at the specified offset and is of the specified size </summary>
        /// <param name="offset"> The starting index of the subvector </param>
        /// <param name="size"> The size of the subvector </param>
        /// <returns> A Vector instance that starts at the data specified by the offset and is of the specified size </returns>
        Vector SubVector(Scalar offset, int size, int stride = 1) const;

        /// <summary> Creates a new Vector instance that contains the same data as this instance </summary>
        /// <returns> A new Vector instance that points to a new, distinct memory that contains the same data as this instance </returns>
        Vector Copy() const;

        /// <summary> Returns the number of active elements within the Vector instance </summary>
        /// <returns> The size of the vector </returns>
        size_t Size() const;

        /// <summary> Retrieves the type of data stored in the wrapped Value instance </summary>
        /// <returns> The type </returns>
        ValueType GetType() const;

        void SetName(const std::string& name);
        std::string GetName() const;

        Vector& operator+=(Scalar);
        Vector& operator-=(Scalar);
        Vector& operator*=(Scalar);
        Vector& operator/=(Scalar);

        Vector& operator+=(Vector);
        Vector& operator-=(Vector);

    private:
        friend Vector operator+(Scalar s, Vector v);
        friend Vector operator+(Vector v, Scalar s);
        friend Vector operator+(Vector v1, Vector v2);

        friend Vector operator-(Scalar s, Vector v);
        friend Vector operator-(Vector v, Scalar s);
        friend Vector operator-(Vector v1, Vector v2);

        friend Vector operator*(Scalar s, Vector v);
        friend Vector operator*(Vector v, Scalar s);
        friend Vector operator*(Vector v, Vector u); // elementwise multiply

        friend Vector operator/(Scalar s, Vector v);
        friend Vector operator/(Vector v, Scalar s);
        friend Vector operator/(Vector v, Vector u); // elementwise divide

        Value _value;
    };

    /// <summary> Constructs an allocated instance of the specified size </summary>
    /// <param name="type"> The type of the elements </param>
    /// <param name="name"> The optional name </param>
    // TODO: Make the type the first param (or investigate why we can't)
    inline Vector MakeVector(int64_t size, ValueType type, const std::string& name = "")
    {
        return Vector(Allocate(type, size), name);
    }

    /// <summary> Constructs an allocated instance of the specified size </summary>
    /// <typeparam name="T"> Any fundamental type accepted by Value </typeparam>
    /// <param name="size"> The size of the allocated vector </param>
    /// <param name="name"> The optional name </param>
    template <typename T>
    Vector MakeVector(int64_t size, const std::string& name = "")
    {
        return Vector(Allocate<T>(size), name);
    }

    /// <summary> Given a view with contiguous memory, return a Vector that represents the entirety of the memory </summary>
    /// <param name="view"> The view that should be reinterpreted as a Vector </param>
    /// <returns> A Vector that encompasses the entire memory region </returns>
    Vector AsVector(ViewAdapter view);

} // namespace value
} // namespace assera
