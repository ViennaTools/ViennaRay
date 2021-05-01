#include <rtTestAsserts.hpp>
#include <rtTrace.hpp>

int main()
{
    constexpr int D = 3;
    using NumericType = double;
    using ParticleType = rtParticle2<NumericType>;
    using ReflectionType = rtReflectionSpecular<NumericType, D>;

    NumericType gridDelta;
    std::vector<rtTriple<NumericType>> points;
    std::vector<rtTriple<NumericType>> normals;
    rtInternal::readGridFromFile("./../Resources/sphereGrid3D_R1.dat", gridDelta, points, normals);

    rtTrace<NumericType, ParticleType, ReflectionType, D> rayTracer;
    rayTracer.setGeometry(points, normals, gridDelta);
    rayTracer.apply();
    rayTracer.setGeometry(points, normals, gridDelta);

    return 0;
}