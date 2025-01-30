#include <rayReflection.hpp>

#include <fstream>

using namespace viennaray;

int main() {

  RNG rngState;

  {
    // diffuse reflection
    Vec3D<double> normal = {0.0, 0.0, 1.0};
    std::ofstream file("diffuse_reflection.txt");
    for (int i = 0; i < 1000; ++i) {
      Vec3D<double> dir = ReflectionDiffuse<double, 3>(normal, rngState);
      file << dir[0] << " " << dir[1] << " " << dir[2] << std::endl;
    }
    file.close();
  }

  Vec3D<double> normal = {0.0, 0.0, 1.0};
  double const minAvgConeAngle = M_PI_2 / 5;
  double const minAngle = 1.3962634;
  Vec3D<double> rayDir = {0.0, -1.0, -1.0};
  Normalize(rayDir);

  auto cosTheta = -DotProduct(rayDir, normal);
  std::cout << "cosTheta: " << cosTheta << std::endl;
  auto incAngle = std::acos(cosTheta);
  std::cout << "incAngle: " << incAngle << std::endl;
  std::cout << "incAngle [deg]: " << incAngle * 180.0 / M_PI << std::endl;
  auto coneAngle = M_PI_2 - std::min(incAngle, minAngle);
  //   auto coneAngle = std::max(incAngle, minAngle);

  {
    // coned specular reflection
    std::ofstream file("coned_specular_reflection.txt");
    for (int i = 0; i < 1000; ++i) {
      Vec3D<double> dir =
          ReflectionConedCosine<double, 3>(rayDir, normal, rngState, coneAngle);
      file << dir[0] << " " << dir[1] << " " << dir[2] << std::endl;
    }
    file.close();
  }

  {
    // coned specular reflection old
    std::ofstream file("coned_specular_reflection_old.txt");
    for (int i = 0; i < 1000; ++i) {
      Vec3D<double> dir = rayInternal::ReflectionConedCosineOld<double, 3>(
          coneAngle, rayDir, normal, rngState);
      file << dir[0] << " " << dir[1] << " " << dir[2] << std::endl;
    }
    file.close();
  }

  return 0;
}