#ifndef RT_UTIL_HPP
#define RT_UTIL_HPP

#include <array>

template <typename NumericType>
using rtPair = std::array<NumericType, 2>;

template <typename NumericType>
using rtTriple = std::array<NumericType, 3>;

template <typename NumericType>
using rtQuadruple = std::array<NumericType, 4>;

#endif // RT_UTIL_HPP