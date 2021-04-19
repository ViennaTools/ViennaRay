#pragma once
#include <cmath>
#include <rtUtil.hpp>

#define RAYTEST_ASSERT( condition )                                     \
{                                                                       \
  if( !( condition ) )                                                  \
  {                                                                     \
    throw std::runtime_error(   std::string( __FILE__ )                 \
                              + std::string( ":" )                      \
                              + std::to_string( __LINE__ )              \
                              + std::string( " in " )                   \
                              + std::string( __PRETTY_FUNCTION__ )      \
                              + std::string(" Condition not fulfilled") \
    );                                                                  \
  }                                                                     \
}

#define RAYTEST_ASSERT_ISCLOSE( first, second, eps )                    \
{                                                                       \
  if( (std::fabs(first - second) > eps) )                               \
  {                                                                     \
    throw std::runtime_error(   std::string( __FILE__ )                 \
                              + std::string( ":" )                      \
                              + std::to_string( __LINE__ )              \
                              + std::string( " in " )                   \
                              + std::string( __PRETTY_FUNCTION__ )      \
                              + std::string(" Number not close")        \
    );                                                                  \
  }                                                                     \
}

#define RAYTEST_ASSERT_ISNORMAL( first, second, eps )                   \
{                                                                       \
  if( (std::fabs(rtInternal::DotProduct(first, second)) > eps) )      \
  {                                                                     \
    throw std::runtime_error(   std::string( __FILE__ )                 \
                              + std::string( ":" )                      \
                              + std::to_string( __LINE__ )              \
                              + std::string( " in " )                   \
                              + std::string( __PRETTY_FUNCTION__ )      \
                              + std::string(" Vectors not normal")      \
    );                                                                  \
  }                                                                     \
}                                                                   
