#pragma once

#define CLASS_ASSERT_NUMERIC_TYPE(ClassName, NumericType)                      \
  static_assert(std::is_floating_point<NumericType>::value,                    \
                #ClassName " only supports floating point numeric types");

#define CLASS_ASSERT_DIMENSION(ClassName, D)                                   \
  static_assert((D == 2) || (D == 3),                                          \
                #ClassName " only supports dimension 2 or 3");

#define CLASS_ASSERT_NUMERIC_TYPE_DIMENSION(ClassName, NumericType, D)         \
  CLASS_ASSERT_NUMERIC_TYPE(ClassName, NumericType)                            \
  CLASS_ASSERT_DIMENSION(ClassName, D)

#define HEADER_INSTANTIATE_TEMPLATE_CLASS_NT(ClassName)                        \
  extern template class ClassName<float>;                                      \
  extern template class ClassName<double>;

#define HEADER_INSTANTIATE_TEMPLATE_CLASS_NT_D(ClassName)                      \
  extern template class ClassName<float, 2>;                                   \
  extern template class ClassName<float, 3>;                                   \
  extern template class ClassName<double, 2>;                                  \
  extern template class ClassName<double, 3>;

#define SOURCE_INSTANTIATE_TEMPLATE_CLASS_NT(ClassName)                        \
  template class ClassName<float>;                                             \
  template class ClassName<double>;

#define SOURCE_INSTANTIATE_TEMPLATE_CLASS_NT_D(ClassName)                      \
  template class ClassName<float, 2>;                                          \
  template class ClassName<float, 3>;                                          \
  template class ClassName<double, 2>;                                         \
  template class ClassName<double, 3>;
