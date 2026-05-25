
#include "gelu_v2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t UB_BLOCK_NUM = 1024;
constexpr uint32_t BYTES_FP16 = 2;
constexpr uint32_t BYTES_FP32 = 4;
constexpr uint32_t MIN_BLOCK_UNITS = 8;
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  GeluV2TilingData tiling;
  uint32_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();

  auto dt = context->GetInputDesc(0)->GetDataType();
  uint32_t dataTypeSize = (dt == ge::DT_FLOAT) ? BYTES_FP32 : BYTES_FP16;
  uint32_t alignNum = BLOCK_BYTES / dataTypeSize;

  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint64_t ubSize = 0;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  uint32_t aivNum = ascendcPlatform.GetCoreNum();

  uint32_t blockSize = UB_BLOCK_NUM * alignNum;
  uint32_t maxBlockByUb = static_cast<uint32_t>((ubSize / BLOCK_BYTES / 12) / MIN_BLOCK_UNITS * MIN_BLOCK_UNITS);
  if (maxBlockByUb > 0) {
    blockSize = maxBlockByUb * alignNum;
  }
  if (blockSize < alignNum) {
    blockSize = alignNum;
  }

  aivNum = (totalLength / blockSize < aivNum) ? (totalLength / blockSize) : aivNum;
  aivNum = aivNum >= 1 ? aivNum : 1;

  uint32_t coreSize = (totalLength / aivNum) / alignNum * alignNum;
  uint32_t coreRemain = totalLength - aivNum * coreSize;

  uint32_t approximate = 0;
  const gert::RuntimeAttrs* attrs = context->GetAttrs();
  if (attrs != nullptr) {
    const int64_t* approximatePtr = attrs->GetInt(0);
    if (approximatePtr != nullptr) {
      approximate = (*approximatePtr == 0) ? 0 : 1;
    }
  }

  tiling.set_totalLength(totalLength);
  tiling.set_alignNum(alignNum);
  tiling.set_blockSize(blockSize);
  tiling.set_coreSize(coreSize);
  tiling.set_coreRemain(coreRemain);
  tiling.set_approximate(approximate);
  context->SetBlockDim(aivNum);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  size_t* currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = 0;

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class GeluV2 : public OpDef {
public:
    explicit GeluV2(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("approximate").AttrType(OPTIONAL).Int(0);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(GeluV2);
}
