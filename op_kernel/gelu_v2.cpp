#include "kernel_operator.h"
#include <type_traits>

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr float GELU_MIN = -5.0f;
constexpr float GELU_MAX = 5.0f;
constexpr float GELU_STEP_INV = 100.0f;
constexpr float GELU_TANH_COEF = 0.7978845608028654f;
constexpr float GELU_TANH_CUBE_COEF = 0.044715f;

static constexpr float GELU_COEFF_POS_LUT[501] = {
    5.000000000e-01f, 5.039893563e-01f, 5.079783137e-01f, 5.119664734e-01f, 5.159534369e-01f, 5.199388058e-01f, 5.239221827e-01f, 5.279031702e-01f,
    5.318813720e-01f, 5.358563926e-01f, 5.398278373e-01f, 5.437953125e-01f, 5.477584260e-01f, 5.517167867e-01f, 5.556700048e-01f, 5.596176924e-01f,
    5.635594629e-01f, 5.674949317e-01f, 5.714237159e-01f, 5.753454347e-01f, 5.792597094e-01f, 5.831661635e-01f, 5.870644226e-01f, 5.909541151e-01f,
    5.948348717e-01f, 5.987063257e-01f, 6.025681132e-01f, 6.064198732e-01f, 6.102612476e-01f, 6.140918812e-01f, 6.179114222e-01f, 6.217195218e-01f,
    6.255158347e-01f, 6.293000189e-01f, 6.330717360e-01f, 6.368306512e-01f, 6.405764332e-01f, 6.443087548e-01f, 6.480272924e-01f, 6.517317265e-01f,
    6.554217416e-01f, 6.590970262e-01f, 6.627572732e-01f, 6.664021794e-01f, 6.700314463e-01f, 6.736447797e-01f, 6.772418897e-01f, 6.808224912e-01f,
    6.843863035e-01f, 6.879330506e-01f, 6.914624613e-01f, 6.949742691e-01f, 6.984682125e-01f, 7.019440346e-01f, 7.054014838e-01f, 7.088403132e-01f,
    7.122602812e-01f, 7.156611510e-01f, 7.190426911e-01f, 7.224046752e-01f, 7.257468822e-01f, 7.290690962e-01f, 7.323711065e-01f, 7.356527079e-01f,
    7.389137003e-01f, 7.421538892e-01f, 7.453730853e-01f, 7.485711049e-01f, 7.517477695e-01f, 7.549029063e-01f, 7.580363478e-01f, 7.611479319e-01f,
    7.642375022e-01f, 7.673049077e-01f, 7.703500028e-01f, 7.733726476e-01f, 7.763727076e-01f, 7.793500537e-01f, 7.823045624e-01f, 7.852361158e-01f,
    7.881446014e-01f, 7.910299121e-01f, 7.938919464e-01f, 7.967306082e-01f, 7.995458067e-01f, 8.023374569e-01f, 8.051054787e-01f, 8.078497979e-01f,
    8.105703452e-01f, 8.132670570e-01f, 8.159398747e-01f, 8.185887451e-01f, 8.212136204e-01f, 8.238144578e-01f, 8.263912197e-01f, 8.289438737e-01f,
    8.314723925e-01f, 8.339767539e-01f, 8.364569407e-01f, 8.389129405e-01f, 8.413447461e-01f, 8.437523550e-01f, 8.461357696e-01f, 8.484949972e-01f,
    8.508300497e-01f, 8.531409436e-01f, 8.554277003e-01f, 8.576903456e-01f, 8.599289099e-01f, 8.621434280e-01f, 8.643339391e-01f, 8.665004868e-01f,
    8.686431190e-01f, 8.707618878e-01f, 8.728568494e-01f, 8.749280644e-01f, 8.769755969e-01f, 8.789995156e-01f, 8.809998925e-01f, 8.829768040e-01f,
    8.849303298e-01f, 8.868605536e-01f, 8.887675626e-01f, 8.906514476e-01f, 8.925123029e-01f, 8.943502263e-01f, 8.961653189e-01f, 8.979576849e-01f,
    8.997274320e-01f, 9.014746710e-01f, 9.031995154e-01f, 9.049020822e-01f, 9.065824910e-01f, 9.082408643e-01f, 9.098773275e-01f, 9.114920086e-01f,
    9.130850381e-01f, 9.146565492e-01f, 9.162066776e-01f, 9.177355613e-01f, 9.192433408e-01f, 9.207301585e-01f, 9.221961595e-01f, 9.236414905e-01f,
    9.250663005e-01f, 9.264707404e-01f, 9.278549630e-01f, 9.292191230e-01f, 9.305633767e-01f, 9.318878820e-01f, 9.331927987e-01f, 9.344782879e-01f,
    9.357445122e-01f, 9.369916355e-01f, 9.382198233e-01f, 9.394292420e-01f, 9.406200594e-01f, 9.417924444e-01f, 9.429465668e-01f, 9.440825975e-01f,
    9.452007083e-01f, 9.463010719e-01f, 9.473838615e-01f, 9.484492515e-01f, 9.494974165e-01f, 9.505285320e-01f, 9.515427737e-01f, 9.525403182e-01f,
    9.535213421e-01f, 9.544860227e-01f, 9.554345372e-01f, 9.563670635e-01f, 9.572837792e-01f, 9.581848624e-01f, 9.590704910e-01f, 9.599408431e-01f,
    9.607960967e-01f, 9.616364296e-01f, 9.624620197e-01f, 9.632730443e-01f, 9.640696809e-01f, 9.648521064e-01f, 9.656204976e-01f, 9.663750306e-01f,
    9.671158813e-01f, 9.678432252e-01f, 9.685572370e-01f, 9.692580911e-01f, 9.699459610e-01f, 9.706210200e-01f, 9.712834402e-01f, 9.719333933e-01f,
    9.725710503e-01f, 9.731965811e-01f, 9.738101551e-01f, 9.744119405e-01f, 9.750021049e-01f, 9.755808147e-01f, 9.761482357e-01f, 9.767045322e-01f,
    9.772498681e-01f, 9.777844056e-01f, 9.783083062e-01f, 9.788217304e-01f, 9.793248371e-01f, 9.798177846e-01f, 9.803007296e-01f, 9.807738278e-01f,
    9.812372336e-01f, 9.816911001e-01f, 9.821355794e-01f, 9.825708221e-01f, 9.829969774e-01f, 9.834141933e-01f, 9.838226166e-01f, 9.842223926e-01f,
    9.846136652e-01f, 9.849965770e-01f, 9.853712692e-01f, 9.857378816e-01f, 9.860965525e-01f, 9.864474189e-01f, 9.867906162e-01f, 9.871262786e-01f,
    9.874545386e-01f, 9.877755273e-01f, 9.880893746e-01f, 9.883962085e-01f, 9.886961558e-01f, 9.889893417e-01f, 9.892758900e-01f, 9.895559229e-01f,
    9.898295613e-01f, 9.900969244e-01f, 9.903581301e-01f, 9.906132945e-01f, 9.908625325e-01f, 9.911059574e-01f, 9.913436810e-01f, 9.915758136e-01f,
    9.918024641e-01f, 9.920237397e-01f, 9.922397464e-01f, 9.924505886e-01f, 9.926563690e-01f, 9.928571893e-01f, 9.930531492e-01f, 9.932443474e-01f,
    9.934308809e-01f, 9.936128452e-01f, 9.937903347e-01f, 9.939634419e-01f, 9.941322583e-01f, 9.942968737e-01f, 9.944573766e-01f, 9.946138540e-01f,
    9.947663918e-01f, 9.949150743e-01f, 9.950599842e-01f, 9.952012034e-01f, 9.953388120e-01f, 9.954728889e-01f, 9.956035117e-01f, 9.957307566e-01f,
    9.958546986e-01f, 9.959754115e-01f, 9.960929674e-01f, 9.962074377e-01f, 9.963188920e-01f, 9.964273990e-01f, 9.965330262e-01f, 9.966358396e-01f,
    9.967359042e-01f, 9.968332837e-01f, 9.969280408e-01f, 9.970202368e-01f, 9.971099319e-01f, 9.971971854e-01f, 9.972820551e-01f, 9.973645979e-01f,
    9.974448697e-01f, 9.975229250e-01f, 9.975988175e-01f, 9.976725998e-01f, 9.977443233e-01f, 9.978140385e-01f, 9.978817950e-01f, 9.979476410e-01f,
    9.980116241e-01f, 9.980737909e-01f, 9.981341867e-01f, 9.981928562e-01f, 9.982498431e-01f, 9.983051900e-01f, 9.983589388e-01f, 9.984111304e-01f,
    9.984618048e-01f, 9.985110013e-01f, 9.985587581e-01f, 9.986051128e-01f, 9.986501020e-01f, 9.986937616e-01f, 9.987361266e-01f, 9.987772313e-01f,
    9.988171093e-01f, 9.988557932e-01f, 9.988933150e-01f, 9.989297061e-01f, 9.989649970e-01f, 9.989992175e-01f, 9.990323968e-01f, 9.990645633e-01f,
    9.990957448e-01f, 9.991259685e-01f, 9.991552608e-01f, 9.991836477e-01f, 9.992111543e-01f, 9.992378053e-01f, 9.992636247e-01f, 9.992886360e-01f,
    9.993128621e-01f, 9.993363251e-01f, 9.993590470e-01f, 9.993810489e-01f, 9.994023515e-01f, 9.994229750e-01f, 9.994429389e-01f, 9.994622626e-01f,
    9.994809646e-01f, 9.994990631e-01f, 9.995165759e-01f, 9.995335201e-01f, 9.995499128e-01f, 9.995657701e-01f, 9.995811081e-01f, 9.995959422e-01f,
    9.996102876e-01f, 9.996241591e-01f, 9.996375709e-01f, 9.996505369e-01f, 9.996630707e-01f, 9.996751856e-01f, 9.996868943e-01f, 9.996982094e-01f,
    9.997091429e-01f, 9.997197067e-01f, 9.997299123e-01f, 9.997397708e-01f, 9.997492931e-01f, 9.997584897e-01f, 9.997673709e-01f, 9.997759467e-01f,
    9.997842266e-01f, 9.997922202e-01f, 9.997999365e-01f, 9.998073844e-01f, 9.998145726e-01f, 9.998215094e-01f, 9.998282029e-01f, 9.998346610e-01f,
    9.998408914e-01f, 9.998469015e-01f, 9.998526985e-01f, 9.998582894e-01f, 9.998636810e-01f, 9.998688798e-01f, 9.998738924e-01f, 9.998787248e-01f,
    9.998833830e-01f, 9.998878730e-01f, 9.998922003e-01f, 9.998963704e-01f, 9.999003886e-01f, 9.999042601e-01f, 9.999079899e-01f, 9.999115827e-01f,
    9.999150433e-01f, 9.999183762e-01f, 9.999215858e-01f, 9.999246764e-01f, 9.999276520e-01f, 9.999305166e-01f, 9.999332742e-01f, 9.999359284e-01f,
    9.999384828e-01f, 9.999409411e-01f, 9.999433065e-01f, 9.999455823e-01f, 9.999477718e-01f, 9.999498779e-01f, 9.999519037e-01f, 9.999538519e-01f,
    9.999557255e-01f, 9.999575271e-01f, 9.999592592e-01f, 9.999609244e-01f, 9.999625251e-01f, 9.999640637e-01f, 9.999655424e-01f, 9.999669634e-01f,
    9.999683288e-01f, 9.999696406e-01f, 9.999709009e-01f, 9.999721116e-01f, 9.999732744e-01f, 9.999743912e-01f, 9.999754636e-01f, 9.999764934e-01f,
    9.999774821e-01f, 9.999784313e-01f, 9.999793425e-01f, 9.999802170e-01f, 9.999810564e-01f, 9.999818618e-01f, 9.999826347e-01f, 9.999833762e-01f,
    9.999840876e-01f, 9.999847700e-01f, 9.999854245e-01f, 9.999860523e-01f, 9.999866543e-01f, 9.999872315e-01f, 9.999877849e-01f, 9.999883154e-01f,
    9.999888240e-01f, 9.999893115e-01f, 9.999897787e-01f, 9.999902264e-01f, 9.999906553e-01f, 9.999910663e-01f, 9.999914601e-01f, 9.999918373e-01f,
    9.999921985e-01f, 9.999925445e-01f, 9.999928759e-01f, 9.999931931e-01f, 9.999934969e-01f, 9.999937877e-01f, 9.999940660e-01f, 9.999943325e-01f,
    9.999945875e-01f, 9.999948315e-01f, 9.999950650e-01f, 9.999952883e-01f, 9.999955021e-01f, 9.999957065e-01f, 9.999959020e-01f, 9.999960890e-01f,
    9.999962678e-01f, 9.999964388e-01f, 9.999966023e-01f, 9.999967586e-01f, 9.999969080e-01f, 9.999970508e-01f, 9.999971873e-01f, 9.999973177e-01f,
    9.999974423e-01f, 9.999975614e-01f, 9.999976751e-01f, 9.999977838e-01f, 9.999978875e-01f, 9.999979867e-01f, 9.999980813e-01f, 9.999981717e-01f,
    9.999982580e-01f, 9.999983403e-01f, 9.999984190e-01f, 9.999984940e-01f, 9.999985656e-01f, 9.999986340e-01f, 9.999986992e-01f, 9.999987614e-01f,
    9.999988208e-01f, 9.999988774e-01f, 9.999989314e-01f, 9.999989829e-01f, 9.999990320e-01f, 9.999990789e-01f, 9.999991235e-01f, 9.999991661e-01f,
    9.999992067e-01f, 9.999992453e-01f, 9.999992822e-01f, 9.999993173e-01f, 9.999993508e-01f, 9.999993827e-01f, 9.999994131e-01f, 9.999994420e-01f,
    9.999994696e-01f, 9.999994958e-01f, 9.999995208e-01f, 9.999995446e-01f, 9.999995673e-01f, 9.999995889e-01f, 9.999996094e-01f, 9.999996289e-01f,
    9.999996475e-01f, 9.999996652e-01f, 9.999996821e-01f, 9.999996981e-01f, 9.999997133e-01f
};

template <typename T>
class KernelGeluV2 {
public:
    __aicore__ inline KernelGeluV2() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t alignNum,
                                uint32_t blockSize, uint32_t coreSize, uint32_t coreRemain,
                                uint32_t approximate)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->totalLength = totalLength;
        this->alignNum = alignNum;
        this->tileLength = blockSize;
        this->approximate = approximate;
        this->blockLength = coreSize + (GetBlockNum() == GetBlockIdx() + 1 ? coreRemain : 0);
        this->blockLength = this->blockLength + (this->blockLength % alignNum ? alignNum - this->blockLength % alignNum : 0);
        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        uint32_t startPointer = coreSize * GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ T*)x + startPointer, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ T*)y + startPointer, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(fxBuf, this->tileLength * sizeof(float));
        pipe.InitBuffer(fyBuf, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmp1Buf, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmp2Buf, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->tileNum - 1; ++i) {
            CopyIn(i, this->tileLength);
            Compute(this->tileLength);
            CopyOut(i, this->tileLength);
        }
        uint32_t lastLength = this->blockLength - this->tileLength * (this->tileNum - 1);
        CopyIn(this->tileNum - 1, lastLength);
        Compute(lastLength);
        CopyOut(this->tileNum - 1, lastLength);
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t length)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopy(xLocal, xGm[progress * this->tileLength], length);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t length)
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        LocalTensor<float> fx = fxBuf.Get<float>();
        LocalTensor<float> fy = fyBuf.Get<float>();

        Cast(fx, xLocal, RoundMode::CAST_NONE, length);
        ComputeTanhApprox(fx, fy, length);
        Cast(yLocal, fy, RoundMode::CAST_NONE, length);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeTanhApprox(LocalTensor<float>& fx, LocalTensor<float>& fy, uint32_t length)
    {
        LocalTensor<float> tmp1 = tmp1Buf.Get<float>();
        LocalTensor<float> tmp2 = tmp2Buf.Get<float>();

        Mul(tmp1, fx, fx, length);
        Mul(tmp1, tmp1, fx, length);
        Muls(tmp1, tmp1, GELU_TANH_CUBE_COEF, length);
        Add(tmp1, tmp1, fx, length);
        Muls(tmp1, tmp1, GELU_TANH_COEF * 2.0f, length);
        Exp(tmp1, tmp1, length);
        Adds(tmp1, tmp1, 1.0f, length);
        Duplicate(tmp2, 2.0f, length);
        Div(tmp2, tmp2, tmp1, length);
        Muls(tmp2, tmp2, -1.0f, length);
        Adds(tmp2, tmp2, 2.0f, length);
        Mul(fy, fx, tmp2, length);
        Muls(fy, fy, 0.5f, length);
    }

    __aicore__ inline float LookupCoeff(float x)
    {
        if (x < GELU_MIN) {
            return 0.0f;
        }
        if (x > GELU_MAX) {
            return 1.0f;
        }

        bool negative = x < 0.0f;
        float ax = negative ? -x : x;
        float pos = ax * GELU_STEP_INV;
        int32_t idx = static_cast<int32_t>(pos);
        if (idx >= 500) {
            return negative ? 2.866515719e-07f : 9.999997133e-01f;
        }

        float left = GELU_COEFF_POS_LUT[idx];
        float right = GELU_COEFF_POS_LUT[idx + 1];
        float coeff = left + (right - left) * (pos - static_cast<float>(idx));
        return negative ? (1.0f - coeff) : coeff;
    }

    __aicore__ inline void ComputeLookupApprox(LocalTensor<float>& fx, LocalTensor<float>& fy, uint32_t length)
    {
        for (uint32_t i = 0; i < length; ++i) {
            float x = fx.GetValue(i);
            fy.SetValue(i, x * LookupCoeff(x));
        }
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t length)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        DataCopy(yGm[progress * this->tileLength], yLocal, length);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> fxBuf;
    TBuf<QuePosition::VECCALC> fyBuf;
    TBuf<QuePosition::VECCALC> tmp1Buf;
    TBuf<QuePosition::VECCALC> tmp2Buf;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    uint32_t totalLength;
    uint32_t alignNum;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t approximate;
};

extern "C" __global__ __aicore__ void gelu_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelGeluV2<DTYPE_X> op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.alignNum, tiling_data.blockSize,
            tiling_data.coreSize, tiling_data.coreRemain, tiling_data.approximate);
    op.Process();
}
