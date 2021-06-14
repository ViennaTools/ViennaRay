#include <rayTestAsserts.hpp>
#include <rayUtil.hpp>

int main() {
  using NumericType = float;
  NumericType eps = 1e-6;

  // Sum/Diff
  {
    rayTriple<NumericType> v1 = {1.2, 2.4, 3.6};
    rayTriple<NumericType> v2 = {2.8, 3.6, 4.4};
    rayTriple<NumericType> v3 = {1., 1., 1.};

    auto result = rayInternal::Sum(v1, v2);
    RAYTEST_ASSERT_ISCLOSE(result[0], 4., eps)
    RAYTEST_ASSERT_ISCLOSE(result[1], 6., eps)
    RAYTEST_ASSERT_ISCLOSE(result[2], 8., eps)

    result = rayInternal::Sum(v1, v2, v3);
    RAYTEST_ASSERT_ISCLOSE(result[0], 5., eps)
    RAYTEST_ASSERT_ISCLOSE(result[1], 7., eps)
    RAYTEST_ASSERT_ISCLOSE(result[2], 9., eps)

    result = rayInternal::Diff(v1, v3);
    RAYTEST_ASSERT_ISCLOSE(result[0], 0.2, eps)
    RAYTEST_ASSERT_ISCLOSE(result[1], 1.4, eps)
    RAYTEST_ASSERT_ISCLOSE(result[2], 2.6, eps)
  }

  // Dot/Cross product
  {
    rayTriple<NumericType> v1 = {1., 0., 1.};
    rayTriple<NumericType> v2 = {1., 0., 0.};

    auto dp = rayInternal::DotProduct(v1, v2);
    RAYTEST_ASSERT_ISCLOSE(dp, 1., eps)

    auto cp = rayInternal::CrossProduct(v1, v2);
    RAYTEST_ASSERT_ISCLOSE(cp[0], 0., eps)
    RAYTEST_ASSERT_ISCLOSE(cp[1], 1., eps)
    RAYTEST_ASSERT_ISCLOSE(cp[2], 0., eps)
  }

  // Normalization
  {
    rayTriple<NumericType> v1 = {1., 1., 1.};

    auto norm = rayInternal::Norm(v1);
    RAYTEST_ASSERT_ISCLOSE(norm, 1.73205, eps)

    rayInternal::Normalize(v1);
    norm = rayInternal::Norm(v1);
    RAYTEST_ASSERT_ISCLOSE(norm, 1., eps)
    RAYTEST_ASSERT(rayInternal::IsNormalized(v1))
  }

  // Other
  {
    rayTriple<NumericType> v1 = {1., 1., 1.};
    rayTriple<NumericType> v2 = {2., 1., 1.};
    auto dist = rayInternal::Distance(v1, v2);
    RAYTEST_ASSERT_ISCLOSE(dist, 1., eps)

    rayInternal::Scale(2.f, v1);
    RAYTEST_ASSERT_ISCLOSE(v1[0], 2., eps)
    RAYTEST_ASSERT_ISCLOSE(v1[1], 2., eps)
    RAYTEST_ASSERT_ISCLOSE(v1[2], 2., eps)

    auto inv = rayInternal::Inv(v1);
    RAYTEST_ASSERT_ISCLOSE(inv[0], -2., eps)
    RAYTEST_ASSERT_ISCLOSE(inv[1], -2., eps)
    RAYTEST_ASSERT_ISCLOSE(inv[2], -2., eps)
  }

  {
    rayTriple<NumericType> p1 = {0., 0., 0.};
    rayTriple<NumericType> p2 = {1., 0., 1.};
    rayTriple<NumericType> p3 = {1., 0., 0.};
    rayTriple<rayTriple<NumericType>> coords = {p1, p2, p3};
    auto result = rayInternal::ComputeNormal(coords);
    RAYTEST_ASSERT_ISCLOSE(result[0], 0., eps)
    RAYTEST_ASSERT_ISCLOSE(result[1], 1., eps)
    RAYTEST_ASSERT_ISCLOSE(result[2], 0., eps)
  }

  return 0;
}