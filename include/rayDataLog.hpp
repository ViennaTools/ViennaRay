#ifndef RAY_DATA_LOG
#define RAY_DATA_LOG

#include <cassert>
#include <vector>

template <class NumericType> struct rayDataLog {

  std::vector<std::vector<NumericType>> data;

  void merge(rayDataLog<NumericType> &pOther) {
    assert(pOther.data.size() == data.size() &&
           "Size mismatch when merging logs");
    for (size_t i = 0; i < data.size(); i++) {
      assert(pOther.data[i].size() == data[i].size() &&
             "Size mismatch when merging log data");
      for (size_t j = 0; j < data[i].size(); j++) {
        data[i][j] += pOther.data[i][j];
      }
    }
  }
};

#endif // RAY_DATA_LOG