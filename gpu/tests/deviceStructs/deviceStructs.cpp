#include <raygLaunchParams.hpp>
#include <raygPerRayData.hpp>

#include <iostream>

int main() {
  viennaray::gpu::PerRayData prd;
  viennaray::gpu::LaunchParams launchParams;

  std::cout << "Size of PerRayData: " << sizeof(prd) << " bytes" << std::endl;
  std::cout << "Size of LaunchParams: " << sizeof(launchParams) << " bytes"
            << std::endl;

  return 0;
}