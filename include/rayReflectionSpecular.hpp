#ifndef RAY_REFLECTIONSPECULAR_HPP
#define RAY_REFLECTIONSPECULAR_HPP

// #include <rayReflection.hpp>

// template <typename NumericType, int D>
// class rayReflectionSpecular : public rayReflection<NumericType, D> {
// public:
//   rayPair<rayTriple<NumericType>> use(RTCRay &rayin, RTCHit &hitin,
//                                       const int materialId,
//                                       rayRNG &RNG) override final {
//     return use(rayin, hitin);
//   }

//   static rayPair<rayTriple<NumericType>> use(RTCRay &rayin, RTCHit &hitin) {
//     auto normal =
//         rayTriple<NumericType>{(NumericType)hitin.Ng_x,
//         (NumericType)hitin.Ng_y,
//                                (NumericType)hitin.Ng_z};
//     rayInternal::Normalize(normal);
//     assert(rayInternal::IsNormalized(normal) &&
//            "rayReflectionSpecular: Surface normal is not normalized");

//     auto dirOldInv = rayInternal::Inv(
//         rayTriple<NumericType>{rayin.dir_x, rayin.dir_y, rayin.dir_z});
//     assert(rayInternal::IsNormalized(dirOldInv) &&
//            "rayReflectionSpecular: Surface normal is not normalized");

//     // Compute new direction
//     auto direction = rayInternal::Diff(
//         rayInternal::Scale(2 * rayInternal::DotProduct(normal, dirOldInv),
//                            normal),
//         dirOldInv);

//     // Compute new origin
//     auto xx = rayin.org_x + rayin.dir_x * rayin.tfar;
//     auto yy = rayin.org_y + rayin.dir_y * rayin.tfar;
//     auto zz = rayin.org_z + rayin.dir_z * rayin.tfar;

//     return {xx, yy, zz, direction};
//   }
// };

#endif // RAY_REFLECTIONSPECULAR_HPP