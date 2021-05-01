#include <omp.h>
#include <rtTrace.hpp>
#include <rtParticle.hpp>
#include <rtReflectionSpecular.hpp>

int main()
{
    constexpr int D = 2;

    using NumericType = float;
    using ParticleType = rtParticle2<NumericType>;
    using ReflectionType = rtReflectionSpecular<NumericType, D>;

    omp_set_num_threads(12);

    NumericType eps = 1e-4;
    NumericType gridDelta;
    std::vector<rtTriple<NumericType>> points;
    std::vector<rtTriple<NumericType>> normals;
    rtInternal::readGridFromFile("./../Resources/trenchGrid2D.dat", gridDelta, points, normals);

    rtTrace<NumericType, ParticleType, ReflectionType, D> rayTracer;
    rayTracer.setGeometry(points, normals, gridDelta);
    rayTracer.setNumberOfRaysPerPoint(1000);
    rayTracer.setSourceDirection(rtTraceDirection::POS_Y);
    rayTracer.setCosinePower(2.);
    rayTracer.apply();
}