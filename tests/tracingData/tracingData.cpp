#include <vcPointData.hpp>
#include <vcTestAsserts.hpp>

using namespace viennacore;

int main() {
  using NumericType = float;
  PointData<NumericType> defaultData;

  defaultData.insertNextScalarData(1000, 0, "zeroData");

  VC_TEST_ASSERT(defaultData.getScalarDataLabel(0) == "zeroData");
  VC_TEST_ASSERT(defaultData.getScalarData("zeroData")->size() == 1000);

  defaultData.insertNextScalarData(1, 0, "oneData");

  VC_TEST_ASSERT(defaultData.getScalarDataLabel(1) == "oneData");
  VC_TEST_ASSERT(defaultData.getScalarData("oneData")->size() == 1);

  PointData<NumericType> movedData = std::move(defaultData);

  VC_TEST_ASSERT(defaultData.getScalarData().data() == nullptr);
  VC_TEST_ASSERT(defaultData.getVectorData().data() == nullptr);

  return 0;
}
