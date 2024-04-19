#pragma once

#include <rayUtil.hpp>

template <typename NumericType, int D> class rayGeometry {
  using pointNeighborhoodType = std::vector<std::vector<unsigned int>>;

public:
  template <size_t Dim>
  void initGeometry(RTCDevice &device,
                    std::vector<std::array<NumericType, Dim>> const &points,
                    std::vector<std::array<NumericType, Dim>> const &normals,
                    NumericType const discRadii) {
    static_assert(!(D == 3 && Dim == 2) &&
                  "Setting 2D geometry in 3D trace object");

    assert(points.size() == normals.size() &&
           "rayGeometry: Points/Normals size mismatch");

    // overwriting the geometry without releasing it beforehand causes the old
    // buffer to leak
    releaseGeometry();
    pRtcGeometry_ =
        rtcNewGeometry(device, RTC_GEOMETRY_TYPE_ORIENTED_DISC_POINT);
    assert(rtcGetDeviceError(device) == RTC_ERROR_NONE &&
           "RTC Error: rtcNewGeometry");
    numPoints_ = points.size();

    // The buffer data is managed internally (embree) and automatically freed
    // when the geometry is destroyed.
    pPointBuffer_ = (point_4f_t *)rtcSetNewGeometryBuffer(
        pRtcGeometry_, RTC_BUFFER_TYPE_VERTEX,
        0, // slot
        RTC_FORMAT_FLOAT4, sizeof(point_4f_t), numPoints_);
    assert(rtcGetDeviceError(device) == RTC_ERROR_NONE &&
           "RTC Error: rtcSetNewGeometryBuffer points");
    discRadii_ = discRadii;

    for (int i = 0; i < D; i++) {
      minCoords_[i] = std::numeric_limits<NumericType>::max();
      maxCoords_[i] = std::numeric_limits<NumericType>::lowest();
    }

    for (size_t i = 0; i < numPoints_; ++i) {
      pPointBuffer_[i].xx = (float)points[i][0];
      pPointBuffer_[i].yy = (float)points[i][1];
      pPointBuffer_[i].radius = (float)discRadii_;
      if (points[i][0] < minCoords_[0])
        minCoords_[0] = points[i][0];
      if (points[i][1] < minCoords_[1])
        minCoords_[1] = points[i][1];
      if (points[i][0] > maxCoords_[0])
        maxCoords_[0] = points[i][0];
      if (points[i][1] > maxCoords_[1])
        maxCoords_[1] = points[i][1];
      if constexpr (D == 2) {
        pPointBuffer_[i].zz = 0.f;
        minCoords_[2] = 0.;
        maxCoords_[2] = 0.;
      } else {
        pPointBuffer_[i].zz = (float)points[i][2];
        if (points[i][2] < minCoords_[2])
          minCoords_[2] = points[i][2];
        if (points[i][2] > maxCoords_[2])
          maxCoords_[2] = points[i][2];
      }
    }

    pNormalVecBuffer_ = (normal_vec_3f_t *)rtcSetNewGeometryBuffer(
        pRtcGeometry_, RTC_BUFFER_TYPE_NORMAL,
        0, // slot
        RTC_FORMAT_FLOAT3, sizeof(normal_vec_3f_t), numPoints_);
    assert(rtcGetDeviceError(device) == RTC_ERROR_NONE &&
           "RTC Error: rtcSetNewGeometryBuffer normals");

    for (size_t i = 0; i < numPoints_; ++i) {
      pNormalVecBuffer_[i].xx = (float)normals[i][0];
      pNormalVecBuffer_[i].yy = (float)normals[i][1];
      if constexpr (D == 2) {
        pNormalVecBuffer_[i].zz = 0.f;
      } else {
        pNormalVecBuffer_[i].zz = (float)normals[i][2];
      }
    }

#ifdef VIENNARAY_USE_RAY_MASKING
    rtcSetGeometryMask(pRtcGeometry_, -1);
#endif

    rtcCommitGeometry(pRtcGeometry_);
    assert(rtcGetDeviceError(device) == RTC_ERROR_NONE &&
           "RTC Error: rtcCommitGeometry");

    initPointNeighborhood(points);
    if (materialIds_.empty()) {
      materialIds_.resize(numPoints_, 0);
    }
  }

  template <typename MatIdType>
  void setMaterialIds(std::vector<MatIdType> const &pMaterialIds) {
    assert(pMaterialIds.size() == numPoints_ &&
           "rayGeometry: Material IDs size mismatch");
    materialIds_.clear();
    materialIds_.reserve(numPoints_);
    for (const auto id : pMaterialIds) {
      materialIds_.push_back(static_cast<int>(id));
    }
  }

  rayPair<rayTriple<NumericType>> getBoundingBox() const {
    return {minCoords_, maxCoords_};
  }

  rayTriple<NumericType> getPoint(const unsigned int primID) const {
    assert(primID < numPoints_ && "rayGeometry: Prim ID out of bounds");
    auto const &pnt = pPointBuffer_[primID];
    return {(NumericType)pnt.xx, (NumericType)pnt.yy, (NumericType)pnt.zz};
  }

  std::vector<unsigned int> const &
  getNeighborIndicies(const unsigned int idx) const {
    assert(idx < numPoints_ && "rayGeometry: Index out of bounds");
    return pointNeighborhood_[idx];
  }

  size_t getNumPoints() const { return numPoints_; }

  NumericType getDiscRadius() const { return discRadii_; }

  RTCGeometry const &getRTCGeometry() const { return pRtcGeometry_; }

  rayTriple<NumericType> getPrimNormal(const unsigned int primID) const {
    assert(primID < numPoints_ && "rayGeometry: Prim ID out of bounds");
    auto const &normal = pNormalVecBuffer_[primID];
    return {(NumericType)normal.xx, (NumericType)normal.yy,
            (NumericType)normal.zz};
  }

  rayQuadruple<rayInternal::rtcNumericType> &getPrimRef(unsigned int primID) {
    assert(primID < numPoints_ && "rayGeometry: Prim ID out of bounds");
    return *reinterpret_cast<rayQuadruple<rayInternal::rtcNumericType> *>(
        &pPointBuffer_[primID]);
  }

  rayTriple<rayInternal::rtcNumericType> &getNormalRef(unsigned int primID) {
    assert(primID < numPoints_ && "rayGeometry: Prim ID out of bounds");
    return *reinterpret_cast<rayTriple<rayInternal::rtcNumericType> *>(
        &pNormalVecBuffer_[primID]);
  }

  int getMaterialId(const unsigned int primID) const {
    assert(primID < numPoints_ && "rayGeometry Prim ID out of bounds");
    return materialIds_[primID];
  }

  bool checkGeometryEmpty() const {
    if (pPointBuffer_ == nullptr || pNormalVecBuffer_ == nullptr ||
        pRtcGeometry_ == nullptr) {
      return true;
    }
    return false;
  }

  void releaseGeometry() {
    // Attention:
    // This function must not be called when the RTCGeometry reference count is
    // > 1. Doing so leads to leaked memory buffers
    if (pPointBuffer_ == nullptr || pNormalVecBuffer_ == nullptr ||
        pRtcGeometry_ == nullptr) {
      return;
    } else {
      rtcReleaseGeometry(pRtcGeometry_);
      pPointBuffer_ = nullptr;
      pNormalVecBuffer_ = nullptr;
      pRtcGeometry_ = nullptr;
    }
  }

private:
  template <size_t Dim>
  void initPointNeighborhood(
      std::vector<std::array<NumericType, Dim>> const &points) {
    pointNeighborhood_.clear();
    pointNeighborhood_.resize(numPoints_, std::vector<unsigned int>{});

    if constexpr (D == 3) {
      std::vector<unsigned int> side1;
      std::vector<unsigned int> side2;

      // create copy of bounding box
      rayTriple<NumericType> min = minCoords_;
      rayTriple<NumericType> max = maxCoords_;

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
      for (unsigned int idx = 0; idx < numPoints_; ++idx) {
        if (points[idx][dirs[dirIdx]] <= pivot) {
          side1.push_back(idx);
        } else {
          side2.push_back(idx);
        }
      }
      createNeighborhood(points, side1, side2, min, max, dirIdx, dirs, pivot);
    } else {
      // TODO: 2D divide and conquer algorithm
      for (unsigned int idx1 = 0; idx1 < numPoints_; ++idx1) {
        for (unsigned int idx2 = idx1 + 1; idx2 < numPoints_; ++idx2) {
          if (checkDistance(points[idx1], points[idx2], 2 * discRadii_)) {
            pointNeighborhood_[idx1].push_back(idx2);
            pointNeighborhood_[idx2].push_back(idx1);
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
    // The pivot element should actually be between min and max.
    if (pivot == min[dirs[dirIdx]] || pivot == max[dirs[dirIdx]]) {
      // In this case the points are extremely close to each other (with respect
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
          pointNeighborhood_[pi1].push_back(pi2);
          pointNeighborhood_[pi2].push_back(pi1);
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
      if (point[dirs[dirIdx]] + 2 * discRadii_ <= pivot) {
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
      if (point[dirs[dirIdx]] - 2 * discRadii_ >= pivot) {
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
                     (4 * discRadii_) &&
                 "Correctness Assertion");
          if (checkDistance(point1, point2, 2 * discRadii_)) {
            pointNeighborhood_[side1Cand[ci1]].push_back(side2Cand[ci2]);
            pointNeighborhood_[side2Cand[ci2]].push_back(side1Cand[ci1]);
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
  static bool checkDistance(const std::array<NumericType, Dim> &p1,
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

private:
  // "RTC_GEOMETRY_TYPE_POINT:
  // The vertex buffer stores each control vertex in the form of a single
  // precision position and radius stored in (x, y, z, r) order in memory
  // (RTC_FORMAT_FLOAT4 format). The number of vertices is inferred from the
  // size of this buffer.
  // Source: https://embree.github.io/api.html#rtc_geometry_type_point
  struct point_4f_t {
    float xx, yy, zz, radius;
  };
  point_4f_t *pPointBuffer_ = nullptr;

  // "RTC_GEOMETRY_TYPE_POINT:
  // [...] the normal buffer stores a single precision normal per control
  // vertex (x, y, z order and RTC_FORMAT_FLOAT3 format)."
  // Source: https://embree.github.io/api.html#rtc_geometry_type_point
  struct normal_vec_3f_t {
    float xx, yy, zz;
  };
  normal_vec_3f_t *pNormalVecBuffer_ = nullptr;

  RTCGeometry pRtcGeometry_ = nullptr;

  size_t numPoints_;
  NumericType discRadii_;
  rayTriple<NumericType> minCoords_;
  rayTriple<NumericType> maxCoords_;
  pointNeighborhoodType pointNeighborhood_;
  std::vector<int> materialIds_;
};
