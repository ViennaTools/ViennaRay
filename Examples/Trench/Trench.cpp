#include <rtTrace.hpp>
#include <rtBoundCondition.hpp>
#include <rtTraceDirection.hpp>
#include <rtReflectionDiffuse.hpp>
#include <omp.h>

template <typename NumericType, int D = 3>
void writeVTK(std::string filename,
              const std::vector<rtTriple<NumericType>> &points,
              const std::vector<NumericType> &mcestimates,
              const std::vector<NumericType> &mcEstNorm)
{
    std::ofstream f(filename.c_str());

    f << "# vtk DataFile Version 2.0" << std::endl;
    f << D << "D Surface" << std::endl;
    f << "ASCII" << std::endl;
    f << "DATASET UNSTRUCTURED_GRID" << std::endl;
    f << "POINTS " << points.size() << " float" << std::endl;

    for (unsigned int i = 0; i < points.size(); i++)
    {
        for (int j = 0; j < 3; j++)
            f << static_cast<float>(points[i][j]) << " ";
        f << std::endl;
    }

    f << "CELLS " << points.size() << " " << points.size() * 2 << std::endl;
    size_t c = 0;
    for (unsigned int i = 0; i < points.size(); i++)
    {
        f << 1 << " " << c++ << std::endl;
    }

    f << "CELL_TYPES " << points.size() << std::endl;
    for (unsigned i = 0; i < points.size(); ++i)
        f << 1 << std::endl;

    f << "CELL_DATA " << mcestimates.size() << std::endl;
    f << "SCALARS mc-estimates float" << std::endl;
    f << "LOOKUP_TABLE default" << std::endl;
    for (unsigned j = 0; j < mcestimates.size(); ++j)
    {
        f << ((std::abs(mcestimates[j]) < 1e-6) ? 0.0 : mcestimates[j]) << std::endl;
    }

    f << "SCALARS mc-est-no-smooth float" << std::endl;
    f << "LOOKUP_TABLE default" << std::endl;
    for (unsigned j = 0; j < mcEstNorm.size(); ++j)
    {
        f << ((std::abs(mcEstNorm[j]) < 1e-6) ? 0.0 : mcEstNorm[j]) << std::endl;
    }

    f.close();
}

int main()
{
    constexpr int D = 3;

    using NumericType = float;
    using ParticleType = rtParticle2<NumericType>;
    using ReflectionType = rtReflectionDiffuse<NumericType, D>;

    omp_set_num_threads(12);

    NumericType gridDelta;
    std::vector<rtTriple<NumericType>> points;
    std::vector<rtTriple<NumericType>> normals;
    rtInternal::readGridFromFile("trenchGrid3D.dat", gridDelta, points, normals);

    rtTraceBoundary bC[D];
    bC[0] = rtTraceBoundary::PERIODIC;
    bC[1] = rtTraceBoundary::PERIODIC;
    bC[2] = rtTraceBoundary::PERIODIC;

    rtTrace<NumericType, ParticleType, ReflectionType, D> rayTracer;
    rayTracer.setGeometry(points, normals, gridDelta);
    rayTracer.setSourceDirection(rtTraceDirection::POS_Z);
    rayTracer.setNumberOfRaysPerPoint(1000);
    rayTracer.setCosinePower(5.);
    rayTracer.setBoundaryConditions(bC);
    rayTracer.apply();

    auto mcEstimates = rayTracer.getMcEstimates();
    auto mcEstNorm = rayTracer.getMcEstimatesNorm();
    writeVTK("trenchResult.vtk", points, mcEstimates, mcEstNorm);

    return 0;
}