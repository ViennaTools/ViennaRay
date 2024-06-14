#include <rayUtil.hpp>
#include <vcTestAsserts.hpp>

using namespace viennaray;

int main() {
  using NumericType = float;
  NumericType eps = 1e-6;

  // Sum/Diff
  {
    Vec3D<NumericType> v1 = {1.2, 2.4, 3.6};
    Vec3D<NumericType> v2 = {2.8, 3.6, 4.4};
    Vec3D<NumericType> v3 = {1., 1., 1.};

    auto result = v1 + v2;
    VC_TEST_ASSERT_ISCLOSE(result[0], 4., eps)
    VC_TEST_ASSERT_ISCLOSE(result[1], 6., eps)
    VC_TEST_ASSERT_ISCLOSE(result[2], 8., eps)

    result = Sum(v1, v2, v3);
    VC_TEST_ASSERT_ISCLOSE(result[0], 5., eps)
    VC_TEST_ASSERT_ISCLOSE(result[1], 7., eps)
    VC_TEST_ASSERT_ISCLOSE(result[2], 9., eps)

    result = v1 - v3;
    VC_TEST_ASSERT_ISCLOSE(result[0], 0.2, eps)
    VC_TEST_ASSERT_ISCLOSE(result[1], 1.4, eps)
    VC_TEST_ASSERT_ISCLOSE(result[2], 2.6, eps)
  }

  // Dot/Cross product
  {
    Vec3D<NumericType> v1 = {1., 0., 1.};
    Vec3D<NumericType> v2 = {1., 0., 0.};

    auto dp = DotProduct(v1, v2);
    VC_TEST_ASSERT_ISCLOSE(dp, 1., eps)

    auto cp = CrossProduct(v1, v2);
    VC_TEST_ASSERT_ISCLOSE(cp[0], 0., eps)
    VC_TEST_ASSERT_ISCLOSE(cp[1], 1., eps)
    VC_TEST_ASSERT_ISCLOSE(cp[2], 0., eps)
  }

  // Normalization
  {
    Vec3D<NumericType> v1 = {1., 1., 1.};

    auto norm = Norm(v1);
    VC_TEST_ASSERT_ISCLOSE(norm, 1.73205, eps)

    Normalize(v1);
    norm = Norm(v1);
    VC_TEST_ASSERT_ISCLOSE(norm, 1., eps)
    VC_TEST_ASSERT(IsNormalized(v1))
  }

  // Other
  {
    Vec3D<NumericType> v1 = {1., 1., 1.};
    Vec3D<NumericType> v2 = {2., 1., 1.};
    auto dist = Distance(v1, v2);
    VC_TEST_ASSERT_ISCLOSE(dist, 1., eps)

    v1 = NumericType(2) * v1;
    VC_TEST_ASSERT_ISCLOSE(v1[0], 2., eps)
    VC_TEST_ASSERT_ISCLOSE(v1[1], 2., eps)
    VC_TEST_ASSERT_ISCLOSE(v1[2], 2., eps)

    auto inv = Inv(v1);
    VC_TEST_ASSERT_ISCLOSE(inv[0], -2., eps)
    VC_TEST_ASSERT_ISCLOSE(inv[1], -2., eps)
    VC_TEST_ASSERT_ISCLOSE(inv[2], -2., eps)
  }

  {
    Vec3D<NumericType> p1 = {0., 0., 0.};
    Vec3D<NumericType> p2 = {1., 0., 1.};
    Vec3D<NumericType> p3 = {1., 0., 0.};
    Vec3D<Vec3D<NumericType>> coords = {p1, p2, p3};
    auto result = ComputeNormal(coords);
    VC_TEST_ASSERT_ISCLOSE(result[0], 0., eps)
    VC_TEST_ASSERT_ISCLOSE(result[1], 1., eps)
    VC_TEST_ASSERT_ISCLOSE(result[2], 0., eps)
  }

  return 0;
}