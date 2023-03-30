#pragma once

#pragma once

#include <raySource.hpp>
#include <rayUtil.hpp>

template <typename NumericType, int D = 2>
class raySourceDistribution : public raySource<NumericType, D>
{
    typedef rayPair<rayTriple<NumericType>> boundingBoxType;

public:
    raySourceDistribution() {}

    raySourceDistribution(boundingBoxType pBoundingBox,
                          const std::array<int, 5> &pTraceSettings)
        : bdBox(pBoundingBox), rayDir(pTraceSettings[0]),
          firstDir(pTraceSettings[1]), secondDir(pTraceSettings[2]),
          minMax(pTraceSettings[3]), posNeg(pTraceSettings[4]),
          mExtent(bdBox[1][firstDir] - bdBox[0][firstDir])
    {
        memset(posPositionDistribution.data(), 0, n_bins_pos);
        memset(negPositionDistribution.data(), 0, n_bins_pos);
        for (size_t i = 0; i < n_bins_pos; i++)
        {
            memset(thetaDistribution[i].data(), 0, n_bins_theta);
        }
    }

    void init(boundingBoxType pBoundingBox,
              const std::array<int, 5> &pTraceSettings)
    {
        bdBox = pBoundingBox;
        rayDir = pTraceSettings[0];
        firstDir = pTraceSettings[1];
        secondDir = pTraceSettings[2];
        minMax = pTraceSettings[3];
        posNeg = pTraceSettings[4];
        mExtent = (bdBox[1][firstDir] - bdBox[0][firstDir]) / 2.;
        memset(posPositionDistribution.data(), 0, n_bins_pos);
        memset(negPositionDistribution.data(), 0, n_bins_pos);
        for (size_t i = 0; i < n_bins_pos; i++)
        {
            memset(thetaDistribution[i].data(), 0, n_bins_theta);
        }
    }

    void fillRay(RTCRay &ray, const size_t idx, rayRNG &RngState1,
                 rayRNG &RngState2, rayRNG &RngState3,
                 rayRNG &RngState4) override final
    {
        auto origin = getOrigin(RngState1);
        auto direction = getDirection(RngState3, origin);

#ifdef ARCH_X86
        reinterpret_cast<__m128 &>(ray) =
            _mm_set_ps(1e-4f, (float)origin[2], (float)origin[1], (float)origin[0]);

        reinterpret_cast<__m128 &>(ray.dir_x) = _mm_set_ps(
            0.0f, (float)direction[2], (float)direction[1], (float)direction[0]);
#else
        ray.org_x = (float)origin[0];
        ray.org_y = (float)origin[1];
        ray.org_z = (float)origin[2];
        ray.tnear = 1e-4f;

        ray.dir_x = (float)direction[0];
        ray.dir_y = (float)direction[1];
        ray.dir_z = (float)direction[2];
        ray.time = 0.0f;
#endif
    }

    void addPosition(const NumericType x, const bool posNeg)
    {
        int bin = x / mExtent * n_bins_pos;
        if (posNeg)
        {
            negPositionDistribution[bin]++;
        }
        else
        {
            posPositionDistribution[bin]++;
        }
    }

    void addTheta(const NumericType x, const bool posNeg,
                  const NumericType theta)
    {
        int thBin = (theta + M_PI_2) / M_PI * n_bins_theta;
        int bin = x / mExtent * n_bins_pos;
        if (!posNeg)
        {
            bin += n_bins_pos;
        }
        thetaDistribution[bin][thBin]++;
    }

    void merge(const raySourceDistribution<NumericType, D> &otherDist)
    {
        for (size_t i = 0; i < n_bins_pos; i++)
        {
            negPositionDistribution[i] += otherDist.negPositionDistribution[i];
            posPositionDistribution[i] += otherDist.posPositionDistribution[i];

            for (size_t j = 0; j < n_bins_theta; j++)
            {
                thetaDistribution[i][j] = otherDist.thetaDistribution[i][j];
                thetaDistribution[i + n_bins_pos][j] =
                    otherDist.thetaDistribution[i + n_bins_pos][j];
            }
        }
    }

    void writeToFile(std::string fileName) const
    {
        std::ofstream file(fileName);
        for (size_t i = 0; i < n_bins_pos; i++)
        {

            file << posPositionDistribution[i] << " " << negPositionDistribution[i]
                 << " ";

            for (size_t j = 0; j < n_bins_theta; j++)
            {
                file << thetaDistribution[i][j] << " ";
            }

            for (size_t j = 0; j < n_bins_theta; j++)
            {
                file << thetaDistribution[i + n_bins_pos][j] << " ";
            }

            file << "\n";
        }
        file.close();
    }

    size_t getNBinsPos() const { return n_bins_pos; }

    size_t getNBinsTheta() const { return n_bins_theta; }

    NumericType getExtent() const { return mExtent; }

private:
    rayTriple<NumericType> getOrigin(rayRNG &RngState)
    {
        rayTriple<NumericType> origin{0., 0., 0.};
        auto r1 = uniDist(RngState);

        origin[rayDir] = bdBox[minMax][rayDir];
        // origin[firstDir] = mRadius * std::sqrt(r1);

        origin[firstDir] =
            bdBox[0][firstDir] + (bdBox[1][firstDir] - bdBox[0][firstDir]) * r1;

        return origin;
    }

    rayTriple<NumericType> getDirection(rayRNG &RngState,
                                        const rayTriple<NumericType> &origin)
    {
        rayTriple<NumericType> direction{0., 0., 0.};

        rayInternal::Normalize(direction);

        return direction;
    }

    boundingBoxType bdBox;
    int rayDir;
    int firstDir;
    int secondDir;
    int minMax;
    NumericType posNeg;
    NumericType mExtent;
    std::uniform_real_distribution<NumericType> uniDist;

    static constexpr size_t n_bins_pos = 100;
    static constexpr size_t n_bins_theta = 100;

    std::array<unsigned, n_bins_pos> posPositionDistribution;
    std::array<unsigned, n_bins_pos> negPositionDistribution;

    std::array<std::array<unsigned, n_bins_theta>, 2 * n_bins_pos>
        thetaDistribution;
};
