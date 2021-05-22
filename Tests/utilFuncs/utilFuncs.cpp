#include <rtTestAsserts.hpp>
#include <rtUtil.hpp>

int main() {
  using NumericType = float;
  NumericType eps = 1e-6;

  // Sum/Diff
  {
    rtTriple<NumericType> v1 = {1.2, 2.4, 3.6};
    rtTriple<NumericType> v2 = {2.8, 3.6, 4.4};
    rtTriple<NumericType> v3 = {1., 1., 1.};

    auto result = rtInternal::Sum(v1, v2);
    RAYTEST_ASSERT_ISCLOSE(result[0], 4., eps)
    RAYTEST_ASSERT_ISCLOSE(result[1], 6., eps)
    RAYTEST_ASSERT_ISCLOSE(result[2], 8., eps)

    result = rtInternal::Sum(v1, v2, v3);
    RAYTEST_ASSERT_ISCLOSE(result[0], 5., eps)
    RAYTEST_ASSERT_ISCLOSE(result[1], 7., eps)
    RAYTEST_ASSERT_ISCLOSE(result[2], 9., eps)

    result = rtInternal::Diff(v1, v3);
    RAYTEST_ASSERT_ISCLOSE(result[0], 0.2, eps)
    RAYTEST_ASSERT_ISCLOSE(result[1], 1.4, eps)
    RAYTEST_ASSERT_ISCLOSE(result[2], 2.6, eps)
  }

  // Dot/Cross product
  {
    rtTriple<NumericType> v1 = {1., 0., 1.};
    rtTriple<NumericType> v2 = {1., 0., 0.};

    auto dp = rtInternal::DotProduct(v1, v2);
    RAYTEST_ASSERT_ISCLOSE(dp, 1., eps)

    auto cp = rtInternal::CrossProduct(v1, v2);
    RAYTEST_ASSERT_ISCLOSE(cp[0], 0., eps)
    RAYTEST_ASSERT_ISCLOSE(cp[1], 1., eps)
    RAYTEST_ASSERT_ISCLOSE(cp[2], 0., eps)
  }

  // Normalization
  {
    rtTriple<NumericType> v1 = {1., 1., 1.};

    auto norm = rtInternal::Norm(v1);
    RAYTEST_ASSERT_ISCLOSE(norm, 1.73205, eps)

    rtInternal::Normalize(v1);
    norm = rtInternal::Norm(v1);
    RAYTEST_ASSERT_ISCLOSE(norm, 1., eps)
    RAYTEST_ASSERT(rtInternal::IsNormalized(v1))
  }

  // Other
  {
    rtTriple<NumericType> v1 = {1., 1., 1.};
    rtTriple<NumericType> v2 = {2., 1., 1.};
    auto dist = rtInternal::Distance(v1, v2);
    RAYTEST_ASSERT_ISCLOSE(dist, 1., eps)

    rtInternal::Scale(2.f, v1);
    RAYTEST_ASSERT_ISCLOSE(v1[0], 2., eps)
    RAYTEST_ASSERT_ISCLOSE(v1[1], 2., eps)
    RAYTEST_ASSERT_ISCLOSE(v1[2], 2., eps)

    auto inv = rtInternal::Inv(v1);
    RAYTEST_ASSERT_ISCLOSE(inv[0], -2., eps)
    RAYTEST_ASSERT_ISCLOSE(inv[1], -2., eps)
    RAYTEST_ASSERT_ISCLOSE(inv[2], -2., eps)
  }

  {
    rtTriple<NumericType> p1 = {0., 0., 0.};
    rtTriple<NumericType> p2 = {1., 0., 1.};
    rtTriple<NumericType> p3 = {1., 0., 0.};
    rtTriple<rtTriple<NumericType>> coords = {p1, p2, p3};
    auto result = rtInternal::ComputeNormal(coords);
    RAYTEST_ASSERT_ISCLOSE(result[0], 0., eps)
    RAYTEST_ASSERT_ISCLOSE(result[1], 1., eps)
    RAYTEST_ASSERT_ISCLOSE(result[2], 0., eps)
  }

  return 0;
}