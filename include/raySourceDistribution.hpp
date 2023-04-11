#pragma once

#include <raySource.hpp>
#include <rayUtil.hpp>

template <typename NumericType, int D = 2>
class raySourceDistribution : public raySource<NumericType, D> {
  typedef rayPair<rayTriple<NumericType>> boundingBoxType;

public:
  raySourceDistribution() {}

  raySourceDistribution(boundingBoxType pBoundingBox,
                        const std::array<int, 5> &pTraceSettings)
      : bdBox(pBoundingBox), rayDir(pTraceSettings[0]),
        firstDir(pTraceSettings[1]), secondDir(pTraceSettings[2]),
        minMax(pTraceSettings[3]), posNeg(pTraceSettings[4]),
        mExtent(bdBox[1][firstDir] - bdBox[0][firstDir]) {
    memset(posPositionDistribution.data(), 0, n_bins_pos);
    memset(negPositionDistribution.data(), 0, n_bins_pos);
    for (size_t i = 0; i < n_bins_pos; i++) {
      memset(thetaDistribution[i].data(), 0, n_bins_theta);
    }
  }

  void init(boundingBoxType pBoundingBox,
            const std::array<int, 5> &pTraceSettings, const bool all = true) {
    bdBox = pBoundingBox;
    rayDir = pTraceSettings[0];
    firstDir = pTraceSettings[1];
    secondDir = pTraceSettings[2];
    minMax = pTraceSettings[3];
    posNeg = pTraceSettings[4];

    if (all) {
      mExtent = (bdBox[1][firstDir] - bdBox[0][firstDir]) / 2.;
      pos_grid_dx = 2 * mExtent / static_cast<NumericType>(n_bins_pos * 2);
      theta_grid_dx = M_PI / static_cast<NumericType>(n_bins_theta);
      for (size_t i = 0; i < n_bins_pos; i++) {
        negPositionDistribution[i] = 0;
        posPositionDistribution[i] = 0;

        for (size_t j = 0; j < n_bins_theta; j++) {
          thetaDistribution[i][j] = 0;
          thetaDistribution[i + n_bins_pos][j] = 0;
        }
      }
    }
  }

  void fillRay(RTCRay &ray, const size_t idx, rayRNG &RngState1,
               rayRNG &RngState2, rayRNG &RngState3,
               rayRNG &RngState4) override final {
    auto origin = getOrigin(RngState1);
    auto direction = getDirection(RngState3);

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

  void addPositionAndTheta(const NumericType x, const bool positionPosNeg,
                           const NumericType theta) {
    assert(theta <= M_PI_2 && theta >= -M_PI_2 && "Invalid theta angle");
    int bin = x / mExtent * n_bins_pos;
    int thBin = (theta + M_PI_2) / M_PI * n_bins_theta;
    if (positionPosNeg) {
      posPositionDistribution[bin]++;
      thetaDistribution[bin + n_bins_pos][thBin]++;
    } else {
      negPositionDistribution[bin]++;
      thetaDistribution[bin][thBin]++;
    }
  }

  void merge(const raySourceDistribution<NumericType, D> &otherDist) {

    for (size_t i = 0; i < n_bins_pos; i++) {
      negPositionDistribution[i] += otherDist.negPositionDistribution[i];
      posPositionDistribution[i] += otherDist.posPositionDistribution[i];

      for (size_t j = 0; j < n_bins_theta; j++) {
        thetaDistribution[i][j] = otherDist.thetaDistribution[i][j];
        thetaDistribution[i + n_bins_pos][j] =
            otherDist.thetaDistribution[i + n_bins_pos][j];
      }
    }
  }

  void writeToFile(std::string fileName) const {
    std::ofstream file(fileName);
    for (size_t i = 0; i < n_bins_pos; i++) {

      file << posPositionDistribution[i] << " " << negPositionDistribution[i]
           << " ";

      for (size_t j = 0; j < n_bins_theta; j++) {
        file << thetaDistribution[i][j] << " ";
      }

      for (size_t j = 0; j < n_bins_theta; j++) {
        file << thetaDistribution[i + n_bins_pos][j] << " ";
      }

      file << "\n";
    }
    file.close();

    std::ofstream fileCsum("Csum_" + fileName);
    for (size_t i = 0; i < 2 * n_bins_pos; i++) {

      fileCsum << posCumSum[i] << " ";

      for (size_t j = 0; j < n_bins_theta; j++) {
        fileCsum << thetaCumSum[i][j] << " ";
      }

      fileCsum << "\n";
    }
    fileCsum.close();
  }

  size_t getNBinsPos() const { return n_bins_pos; }

  size_t getNBinsTheta() const { return n_bins_theta; }

  NumericType getExtent() const { return mExtent; }

  void prepareDistribution() {
    posCumSum[0] = negPositionDistribution[n_bins_pos - 1];
    for (int i = 1; i < n_bins_pos; i++) {
      posCumSum[i] =
          negPositionDistribution[n_bins_pos - i - 1] + posCumSum[i - 1];
    }

    posCumSum[n_bins_pos] = posPositionDistribution[0];
    for (int i = 1; i < n_bins_pos; i++) {
      posCumSum[i + n_bins_pos] =
          posPositionDistribution[i] + posCumSum[i + n_bins_pos - 1];
    }

    for (int i = 0; i < n_bins_pos; i++) {
      thetaCumSum[i][0] = thetaDistribution[n_bins_pos - 1 - i][0];
      for (int j = 1; j < n_bins_theta; j++) {
        thetaCumSum[i][j] =
            thetaCumSum[i][j - 1] + thetaDistribution[n_bins_pos - 1 - i][j];
      }

      thetaCumSum[i + n_bins_pos][0] = thetaDistribution[n_bins_pos + i][0];
      for (int j = 1; j < n_bins_theta; j++) {
        thetaCumSum[i + n_bins_pos][j] = thetaCumSum[i + n_bins_pos][j - 1] +
                                         thetaDistribution[n_bins_pos + i][j];
      }
    }
  }

  std::pair<NumericType, rayPair<NumericType>> getSample(rayRNG &rngState) {
    auto rand = uniDist(rngState) * posCumSum[2 * n_bins_pos - 1];
    std::size_t idx =
        std::lower_bound(posCumSum.begin(), posCumSum.end(), rand) -
        posCumSum.begin();

    NumericType dx_start =
        static_cast<NumericType>(idx) * pos_grid_dx - mExtent;

    rand = uniDist(rngState);
    NumericType position = dx_start + rand * pos_grid_dx;

    rand = uniDist(rngState) * thetaCumSum[idx][n_bins_theta - 1];
    std::size_t theta_idx = std::lower_bound(thetaCumSum[idx].begin(),
                                             thetaCumSum[idx].end(), rand) -
                            thetaCumSum[idx].begin();

    NumericType dt_start =
        static_cast<NumericType>(theta_idx) * theta_grid_dx - M_PI_2;
    rand = uniDist(rngState);
    NumericType theta = dt_start + rand * theta_grid_dx;

    rayPair<NumericType> direction{std::sin(theta), -std::cos(theta)};

    return {position, direction};
  }

  void setNumPoints(const size_t pNumPoints) { numPoints = pNumPoints; }

  size_t getNumPoints() const override { return numPoints; }

  void move(raySourceDistribution<NumericType, 2> &other) {
    posCumSum = std::move(other.posCumSum);
    thetaCumSum = std::move(other.thetaCumSum);

    mExtent = other.mExtent;
    pos_grid_dx = other.pos_grid_dx;
    theta_grid_dx = other.theta_grid_dx;
  }

private:
  rayTriple<NumericType> getOrigin(rayRNG &rngState) {
    rayTriple<NumericType> origin{0., 0., 0.};

    auto rand = uniDist(rngState) * posCumSum[2 * n_bins_pos - 1];
    idx = std::lower_bound(posCumSum.begin(), posCumSum.end(), rand) -
          posCumSum.begin();

    NumericType dx_start =
        static_cast<NumericType>(idx) * pos_grid_dx - mExtent;

    rand = uniDist(rngState);
    NumericType position = dx_start + rand * pos_grid_dx;

    origin[rayDir] = bdBox[minMax][rayDir];
    origin[firstDir] = position;

    return origin;
  }

  rayTriple<NumericType> getDirection(rayRNG &rngState) {
    rayTriple<NumericType> direction{0., 0., 0.};

    auto rand = uniDist(rngState) * thetaCumSum[idx][n_bins_theta - 1];
    std::size_t theta_idx = std::lower_bound(thetaCumSum[idx].begin(),
                                             thetaCumSum[idx].end(), rand) -
                            thetaCumSum[idx].begin();

    NumericType dt_start =
        static_cast<NumericType>(theta_idx) * theta_grid_dx - M_PI_2;
    rand = uniDist(rngState);
    NumericType theta = dt_start + rand * theta_grid_dx;

    direction[firstDir] = std::sin(theta);
    direction[rayDir] = posNeg * std::cos(theta);

    // rayInternal::Normalize(direction);

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
  size_t numPoints = 0;
  size_t idx = 0;

  NumericType pos_grid_dx;
  NumericType theta_grid_dx;

  static constexpr size_t n_bins_pos = 100;
  static constexpr size_t n_bins_theta = 100;

  std::array<unsigned, n_bins_pos> posPositionDistribution;
  std::array<unsigned, n_bins_pos> negPositionDistribution;

  std::array<std::array<unsigned, n_bins_theta>, 2 * n_bins_pos>
      thetaDistribution;

  std::array<NumericType, n_bins_pos * 2> posCumSum;
  std::array<std::array<NumericType, n_bins_theta>, 2 * n_bins_pos> thetaCumSum;
};
