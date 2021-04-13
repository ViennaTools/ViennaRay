#ifndef RT_UTIL_HPP
#define RT_UTIL_HPP

#include <array>
#include <cmath>

template <typename NumericType>
using rtPair = std::array<NumericType, 2>;

template <typename NumericType>
using rtTriple = std::array<NumericType, 3>;

template <typename NumericType>
using rtQuadruple = std::array<NumericType, 4>;

template <typename NumericType>
NumericType rtUtilDistance(const rtTriple<NumericType> &vec1, const rtTriple<NumericType> &vec2)
{
    NumericType d0 = vec1[0] - vec2[0];
    NumericType d1 = vec1[1] - vec2[1];
    NumericType d2 = vec1[2] - vec2[2];
    return std::sqrt(d0 * d0 + d1 * d1 + d2 * d2);
}

template <typename NumericType>
void printTriple(const rtTriple<NumericType> &vec)
{
    std::cout << "(" << vec[0] << ", " << vec[1] << ", " << vec[2] << ")" << std::endl;
}

#endif // RT_UTIL_HPP