#ifndef RAY_GEOMETRY_HPP
#define RAY_GEOMETRY_HPP

#include <embree3/rtcore.h>
#include <rayMessage.hpp>
#include <rayMetaGeometry.hpp>
#include <rayUtil.hpp>

template <typename NumericType, int D>
class rayGeometry : public rayMetaGeometry<NumericType, D> {
private:
  typedef std::vector<std::vector<unsigned int>> pointNeighborhoodType;

public:
  rayGeometry() {}

  template <size_t Dim>
  void initGeometry(RTCDevice &pDevice,
                    std::vector<std::array<NumericType, Dim>> &points,
                    std::vector<std::array<NumericType, Dim>> &normals,
                    NumericType discRadii) {
    static_assert(!(D == 3 && Dim == 2) &&
                  "Setting 2D geometry in 3D trace object");

    assert(points.size() == normals.size() &&
           "rayGeometry: Points/Normals size missmatch");

    // overwriting the geometry without releasing it beforehand causes the old
    // buffer to leak
    releaseGeometry();
    mRTCGeometry =
        rtcNewGeometry(pDevice, RTC_GEOMETRY_TYPE_ORIENTED_DISC_POINT);
    assert(rtcGetDeviceError(pDevice) == RTC_ERROR_NONE &&
           "RTC Error: rtcNewGeometry");
    mNumPoints = points.size();

    // The buffer data is managed internally (embree) and automatically freed
    // when the geometry is destroyed.
    mPointBuffer = (point_4f_t *)rtcSetNewGeometryBuffer(
        mRTCGeometry, RTC_BUFFER_TYPE_VERTEX,
        0, // slot
        RTC_FORMAT_FLOAT4, sizeof(point_4f_t), mNumPoints);
    assert(rtcGetDeviceError(pDevice) == RTC_ERROR_NONE &&
           "RTC Error: rtcSetNewGeometryBuffer points");
    mDiscRadii = discRadii;

    for (size_t i = 0; i < mNumPoints; ++i) {
      mPointBuffer[i].xx = (float)points[i][0];
      mPointBuffer[i].yy = (float)points[i][1];
      mPointBuffer[i].radius = (float)discRadii;
      if (points[i][0] < mMinCoords[0])
        mMinCoords[0] = points[i][0];
      if (points[i][1] < mMinCoords[1])
        mMinCoords[1] = points[i][1];
      if (points[i][0] > mMaxCoords[0])
        mMaxCoords[0] = points[i][0];
      if (points[i][1] > mMaxCoords[1])
        mMaxCoords[1] = points[i][1];
      if constexpr (D == 2) {
        mPointBuffer[i].zz = 0.f;
        mMinCoords[2] = 0.;
        mMaxCoords[2] = 0.;
      } else {
        mPointBuffer[i].zz = (float)points[i][2];
        if (points[i][2] < mMinCoords[2])
          mMinCoords[2] = points[i][2];
        if (points[i][2] > mMaxCoords[2])
          mMaxCoords[2] = points[i][2];
      }
    }

    mNormalVecBuffer = (normal_vec_3f_t *)rtcSetNewGeometryBuffer(
        mRTCGeometry, RTC_BUFFER_TYPE_NORMAL,
        0, // slot
        RTC_FORMAT_FLOAT3, sizeof(normal_vec_3f_t), mNumPoints);
    assert(rtcGetDeviceError(pDevice) == RTC_ERROR_NONE &&
           "RTC Error: rtcSetNewGeometryBuffer normals");

    for (size_t i = 0; i < mNumPoints; ++i) {
      mNormalVecBuffer[i].xx = (float)normals[i][0];
      mNormalVecBuffer[i].yy = (float)normals[i][1];
      if constexpr (D == 2) {
        mNormalVecBuffer[i].zz = 0.f;
      } else {
        mNormalVecBuffer[i].zz = (float)normals[i][2];
      }
    }

    rtcCommitGeometry(mRTCGeometry);
    assert(rtcGetDeviceError(pDevice) == RTC_ERROR_NONE &&
           "RTC Error: rtcCommitGeometry");

    initPointNeighborhood(points);
    if (mMaterialIds.empty()) {
      mMaterialIds.resize(mNumPoints, 0);
    }
  }

  template <typename MatIdType>
  void setMaterialIds(std::vector<MatIdType> &pMaterialIds) {
    assert(pMaterialIds.size() == mNumPoints &&
           "rayGeometry: Material IDs size missmatch");
    mMaterialIds.clear();
    mMaterialIds.reserve(mNumPoints);
    for (const auto id : pMaterialIds) {
      mMaterialIds.push_back(static_cast<int>(id));
    }
  }

  rayPair<rayTriple<NumericType>> getBoundingBox() const {
    return {mMinCoords, mMaxCoords};
  }

  rayTriple<NumericType> getPoint(const unsigned int primID) const {
    assert(primID < mNumPoints && "rayGeometry: Prim ID out of bounds");
    auto const &pnt = mPointBuffer[primID];
    return {(NumericType)pnt.xx, (NumericType)pnt.yy, (NumericType)pnt.zz};
  }

  std::vector<unsigned int> getNeighborIndicies(const unsigned int idx) const {
    assert(idx < mNumPoints && "rayGeometry: Index out of bounds");
    return mPointNeighborhood[idx];
  }

  size_t getNumPoints() const { return mNumPoints; }

  NumericType getDiscRadius() const { return mDiscRadii; }

  RTCGeometry &getRTCGeometry() override final { return mRTCGeometry; }

  rayTriple<NumericType> getPrimNormal(const unsigned int primID) override final {
    assert(primID < mNumPoints && "rayGeometry: Prim ID out of bounds");
    auto const &normal = mNormalVecBuffer[primID];
    return {(NumericType)normal.xx, (NumericType)normal.yy,
            (NumericType)normal.zz};
  }

  rayQuadruple<rtcNumericType> &getPrimRef(unsigned int primID) {
    assert(primID < mNumPoints && "rayGeometry: Prim ID out of bounds");
    return *reinterpret_cast<rayQuadruple<rtcNumericType> *>(
        &mPointBuffer[primID]);
  }

  rayTriple<rtcNumericType> &getNormalRef(unsigned int primID) {
    assert(primID < mNumPoints && "rayGeometry: Prim ID out of bounds");
    return *reinterpret_cast<rayTriple<rtcNumericType> *>(
        &mNormalVecBuffer[primID]);
  }

  std::vector<int> &getMaterialIds() { return mMaterialIds; }

  int getMaterialId(const unsigned int primID) const override final {
    assert(primID < mNumPoints && "rayGeometry Prim ID out of bounds");
    return mMaterialIds[primID];
  }

  bool checkGeometryEmpty() {
    if (mPointBuffer == nullptr || mNormalVecBuffer == nullptr ||
        mRTCGeometry == nullptr) {
      return true;
    }
    return false;
  }

  void releaseGeometry() {
    // Attention:
    // This function must not be called when the RTCGeometry reference count is
    // > 1. Doing so leads to leaked memory buffers
    if (mPointBuffer == nullptr || mNormalVecBuffer == nullptr ||
        mRTCGeometry == nullptr) {
      return;
    } else {
      rtcReleaseGeometry(mRTCGeometry);
      mPointBuffer = nullptr;
      mNormalVecBuffer = nullptr;
      mRTCGeometry = nullptr;
    }
  }

private:
  template <size_t Dim>
  void
  initPointNeighborhood(std::vector<std::array<NumericType, Dim>> &points) {
    mPointNeighborhood.clear();
    mPointNeighborhood.resize(mNumPoints, std::vector<unsigned int>{});

    if constexpr (D == 3) {
      std::vector<unsigned int> side1;
      std::vector<unsigned int> side2;

      // create copy of bounding box
      rayTriple<NumericType> min = mMinCoords;
      rayTriple<NumericType> max = mMaxCoords;

      std::vector<int> dirs;
      for (int i = 0; i < 3; ++i) {
        if (min[i] != max[i]) {
          dirs.push_back(i);
        }
      }
      dirs.shrink_to_fit();

      int dirIdx = 0;
      NumericType pivot = (max[dirs[dirIdx]] + min[dirs[dirIdx]]) / 2;

      // divide point data
      for (unsigned int idx = 0; idx < mNumPoints; ++idx) {
        if (points[idx][dirs[dirIdx]] <= pivot) {
          side1.push_back(idx);
        } else {
          side2.push_back(idx);
        }
      }
      createNeighborhood(points, side1, side2, min, max, dirIdx, dirs, pivot);
    } else {
      // TODO: 2D divide and conquer algorithm
      for (unsigned int idx1 = 0; idx1 < mNumPoints; ++idx1) {
        for (unsigned int idx2 = idx1 + 1; idx2 < mNumPoints; ++idx2) {
          if (checkDistance(points[idx1], points[idx2], 2 * mDiscRadii)) {
            mPointNeighborhood[idx1].push_back(idx2);
            mPointNeighborhood[idx2].push_back(idx1);
          }
        }
      }
    }
  }

  void createNeighborhood(const std::vector<rayTriple<NumericType>> &points,
                          const std::vector<unsigned int> &side1,
                          const std::vector<unsigned int> &side2,
                          const rayTriple<NumericType> &min,
                          const rayTriple<NumericType> &max, const int &dirIdx,
                          const std::vector<int> &dirs,
                          const NumericType &pivot) {
    assert(0 <= dirIdx && dirIdx < dirs.size() && "Assumption");
    if (side1.size() + side2.size() <= 1) {
      return;
    }

    // Corner case
    // The pivot element should actually be inbetween min and max.
    if (pivot == min[dirs[dirIdx]] || pivot == max[dirs[dirIdx]]) {
      // In this case the points are extremly close to each other (with respect
      // to the floating point precision).
      assert((min[dirs[dirIdx]] + max[dirs[dirIdx]]) / 2 == pivot &&
             "Characterization of corner case");
      auto sides = std::vector<unsigned int>(side1);
      sides.insert(sides.end(), side2.begin(), side2.end());
      // Add each of them to the neighborhoods
      for (unsigned int idx1 = 0; idx1 < sides.size() - 1; ++idx1) {
        for (unsigned int idx2 = idx1 + 1; idx2 < sides.size(); ++idx2) {
          auto const &pi1 = sides[idx1];
          auto const &pi2 = sides[idx2];
          assert(pi1 != pi2 && "Assumption");
          mPointNeighborhood[pi1].push_back(pi2);
          mPointNeighborhood[pi2].push_back(pi1);
        }
      }
      return;
    }

    // sets of candidates
    std::vector<unsigned int> side1Cand;
    std::vector<unsigned int> side2Cand;

    int newDirIdx = (dirIdx + 1) % dirs.size();
    NumericType newPivot = (max[dirs[newDirIdx]] + min[dirs[newDirIdx]]) / 2;

    // recursion sets
    std::vector<unsigned int> s1r1set;
    std::vector<unsigned int> s1r2set;
    std::vector<unsigned int> s2r1set;
    std::vector<unsigned int> s2r2set;

    for (unsigned int idx = 0; idx < side1.size(); ++idx) {
      const auto &point = points[side1[idx]];
      assert(point[dirs[dirIdx]] <= pivot && "Correctness Assertion");
      if (point[dirs[newDirIdx]] <= newPivot) {
        s1r1set.push_back(side1[idx]);
      } else {
        s1r2set.push_back(side1[idx]);
      }
      if (point[dirs[dirIdx]] + 2 * mDiscRadii <= pivot) {
        continue;
      }
      side1Cand.push_back(side1[idx]);
    }
    for (unsigned int idx = 0; idx < side2.size(); ++idx) {
      const auto &point = points[side2[idx]];
      assert(point[dirs[dirIdx]] > pivot && "Correctness Assertion");
      if (point[dirs[newDirIdx]] <= newPivot) {
        s2r1set.push_back(side2[idx]);
      } else {
        s2r2set.push_back(side2[idx]);
      }
      if (point[dirs[dirIdx]] - 2 * mDiscRadii >= pivot) {
        continue;
      }
      side2Cand.push_back(side2[idx]);
    }

    // Iterate over pairs of candidates
    if (side1Cand.size() > 0 && side2Cand.size() > 0) {
      for (unsigned int ci1 = 0; ci1 < side1Cand.size(); ++ci1) {
        for (unsigned int ci2 = 0; ci2 < side2Cand.size(); ++ci2) {
          const auto &point1 = points[side1Cand[ci1]];
          const auto &point2 = points[side2Cand[ci2]];

          assert(std::abs(point1[dirs[dirIdx]] - point2[dirs[dirIdx]]) <=
                     (4 * mDiscRadii) &&
                 "Correctness Assertion");
          if (checkDistance(point1, point2, 2 * mDiscRadii)) {
            mPointNeighborhood[side1Cand[ci1]].push_back(side2Cand[ci2]);
            mPointNeighborhood[side2Cand[ci2]].push_back(side1Cand[ci1]);
          }
        }
      }
    }

    // Recurse
    if (side1.size() > 1) {
      auto newS1Max = max;
      newS1Max[dirs[dirIdx]] = pivot; // old diridx and old pivot!
      createNeighborhood(points, s1r1set, s1r2set, min, newS1Max, newDirIdx,
                         dirs, newPivot);
    }
    if (side2.size() > 1) {
      auto newS2Min = min;
      newS2Min[dirs[dirIdx]] = pivot; // old diridx and old pivot!
      createNeighborhood(points, s2r1set, s2r2set, newS2Min, max, newDirIdx,
                         dirs, newPivot);
    }
  }

  template <size_t Dim>
  bool checkDistance(const std::array<NumericType, Dim> &p1,
                     const std::array<NumericType, Dim> &p2,
                     const NumericType &dist) {
    for (int i = 0; i < D; ++i) {
      if (std::abs(p1[i] - p2[i]) >= dist)
        return false;
    }
    if (rayInternal::Distance<NumericType>(p1, p2) < dist)
      return true;

    return false;
  }

  // "RTC_GEOMETRY_TYPE_POINT:
  // The vertex buffer stores each control vertex in the form of a single
  // precision position and radius stored in (x, y, z, r) order in memory
  // (RTC_FORMAT_FLOAT4 format). The number of vertices is inferred from the
  // size of this buffer.
  // Source: https://embree.github.io/api.html#rtc_geometry_type_point
  struct point_4f_t {
    float xx, yy, zz, radius;
  };
  point_4f_t *mPointBuffer = nullptr;

  // "RTC_GEOMETRY_TYPE_POINT:
  // [...] the normal buffer stores a single precision normal per control
  // vertex (x, y, z order and RTC_FORMAT_FLOAT3 format)."
  // Source: https://embree.github.io/api.html#rtc_geometry_type_point
  struct normal_vec_3f_t {
    float xx, yy, zz;
  };
  normal_vec_3f_t *mNormalVecBuffer = nullptr;

  RTCGeometry mRTCGeometry = nullptr;

  size_t mNumPoints;
  NumericType mDiscRadii;
  constexpr static NumericType nummax = std::numeric_limits<NumericType>::max();
  constexpr static NumericType nummin =
      std::numeric_limits<NumericType>::lowest();
  rayTriple<NumericType> mMinCoords{nummax, nummax, nummax};
  rayTriple<NumericType> mMaxCoords{nummin, nummin, nummin};
  pointNeighborhoodType mPointNeighborhood;
  std::vector<int> mMaterialIds;
};

#endif // RAY_GEOMETRY_HPP