#include <rayReflection.hpp>
#include <vcTimer.hpp>

#include <fstream>

// #define VR_WRITE_TO_FILE

using namespace viennaray;

int main() {

  using NumericType = double;
  constexpr int N = 50000;
  RNG rngState(12351263);
  Timer timer;

  {
    std::array<Vec3D<NumericType>, N> directions;

    // diffuse reflection
    Vec3D<NumericType> normal = {0.0, 0.0, 1.0};

    timer.start();
    for (int i = 0; i < N; ++i) {
      directions[i] = ReflectionDiffuse<NumericType, 3>(normal, rngState);
    }
    timer.finish();

    std::cout << "Time for " << N
              << " diffuse reflections: " << timer.currentDuration * 1e-6
              << " ms" << std::endl;

#ifdef VR_WRITE_TO_FILE
    std::ofstream file("diffuse_reflection.txt");
    for (auto const &dir : directions) {
      file << dir[0] << " " << dir[1] << " " << dir[2] << std::endl;
    }
    file.close();
#endif
  }

  Vec3D<NumericType> normal = {0.0, 0.0, 1.0};
  NumericType const minAngle = 85.0 * M_PI / 180.0;
  NumericType incAngle = 75.0 * M_PI / 180.0;
  Vec3D<NumericType> rayDir = {0.0, -std::sin(incAngle), -std::cos(incAngle)};
  auto coneAngle = M_PI_2 - std::min(incAngle, minAngle);

  std::cout << "minAngle: " << minAngle << std::endl;
  std::cout << "incAngle: " << incAngle << std::endl;
  std::cout << "incAngle [deg]: " << incAngle * 180.0 / M_PI << std::endl;
  std::cout << "coneAngle: " << coneAngle << std::endl;

  {
    std::array<Vec3D<NumericType>, N> directions;

    // coned specular reflection
    timer.start();
    for (int i = 0; i < N; ++i) {
      directions[i] = ReflectionConedCosine<NumericType, 3>(
          rayDir, normal, rngState, coneAngle);
    }
    timer.finish();

    std::cout << "Time for " << N
              << " coned specular reflections: " << timer.currentDuration * 1e-6
              << " ms" << std::endl;

#ifdef VR_WRITE_TO_FILE
    std::ofstream file("coned_specular_reflection.txt");
    for (auto const &dir : directions) {
      file << dir[0] << " " << dir[1] << " " << dir[2] << std::endl;
    }
    file.close();
#endif
  }

  {
    std::array<Vec3D<NumericType>, N> directions;

    // coned specular reflection
    timer.start();
    for (int i = 0; i < N; ++i) {
      directions[i] = rayInternal::ReflectionConedCosineOld<NumericType, 3>(
          rayDir, normal, rngState, coneAngle);
    }
    timer.finish();

    std::cout << "Time for " << N << " coned specular reflections (old): "
              << timer.currentDuration * 1e-6 << " ms" << std::endl;

#ifdef VR_WRITE_TO_FILE
    std::ofstream file("coned_specular_reflection_old.txt");
    for (auto const &dir : directions) {
      file << dir[0] << " " << dir[1] << " " << dir[2] << std::endl;
    }
    file.close();
#endif
  }

  return 0;
}