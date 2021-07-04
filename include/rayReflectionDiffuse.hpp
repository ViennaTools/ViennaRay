#ifndef RAY_REFLECTIONDIFFUSE_HPP
#define RAY_REFLECTIONDIFFUSE_HPP

// #include <rayReflection.hpp>

// template <typename NumericType, int D>
// class rayReflectionDiffuse : public rayReflection<NumericType, D> {

// public:
//   rayPair<rayTriple<NumericType>> use(RTCRay &rayin, RTCHit &hitin,
//                                       const int materialId,
//                                       rayRNG &RNG) override final {
//     auto normal =
//         rayTriple<NumericType>{(NumericType)hitin.Ng_x,
//         (NumericType)hitin.Ng_y,
//                                (NumericType)hitin.Ng_z};
//     rayInternal::Normalize(normal);
//     assert(rayInternal::IsNormalized(normal) &&
//            "rayReflectionDiffuse: Surface normal is not normalized");

//     if constexpr (D == 3) {

//       // Compute lambertian reflection with respect to surface normal
//       const auto orthonormalBasis = rayInternal::getOrthonormalBasis(normal);
//       auto newDirection = getCosineHemi(orthonormalBasis, RNG);
//       assert(rayInternal::IsNormalized(newDirection) &&
//              "rayReflectionDiffuse: New direction is not normalized");
//       // Compute new origin
//       auto xx = rayin.org_x + rayin.dir_x * rayin.tfar;
//       auto yy = rayin.org_y + rayin.dir_y * rayin.tfar;
//       auto zz = rayin.org_z + rayin.dir_z * rayin.tfar;

//       return {xx, yy, zz, newDirection};
//     } else {
//       const auto angle =
//           ((NumericType)RNG() / (NumericType)RNG.max() - 0.5) *
//           rayInternal::PI;
//       const auto cos = std::cos(angle);
//       const auto sin = std::sin(angle);
//       auto newDirection =
//           rayTriple<NumericType>{cos * normal[0] - sin * normal[1],
//                                  sin * normal[0] + cos * normal[1], 0.};
//       assert(rayInternal::IsNormalized(newDirection) &&
//              "rayReflectionDiffuse: New direction is not normalized");
//       // Compute new origin
//       auto xx = rayin.org_x + rayin.dir_x * rayin.tfar;
//       auto yy = rayin.org_y + rayin.dir_y * rayin.tfar;

//       return {xx, yy, 0., newDirection};
//     }
//   }

// private:
//   rayTriple<NumericType>
//   getCosineHemi(const rayTriple<rayTriple<NumericType>> &basis, rayRNG &RNG)
//   {
//     std::uniform_real_distribution<NumericType> uniDist;
//     auto r1 = uniDist(RNG);
//     auto r2 = uniDist(RNG);

//     constexpr NumericType two_pi = 2 * rayInternal::PI;
//     NumericType cc1 = sqrt(r2);
//     NumericType cc2 = cos(two_pi * r1) * sqrt(1 - r2);
//     NumericType cc3 = sin(two_pi * r1) * sqrt(1 - r2);

//     auto tt1 = basis[0];
//     rayInternal::Scale(cc1, tt1);
//     auto tt2 = basis[1];
//     rayInternal::Scale(cc2, tt2);
//     auto tt3 = basis[2];
//     rayInternal::Scale(cc3, tt3);

//     return rayInternal::Sum(tt1, tt2, tt3);
//   }
// };

#endif // RAY_REFLECTIONDIFFUSE_HPP