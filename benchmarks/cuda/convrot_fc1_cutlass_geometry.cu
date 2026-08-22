#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cutlass/cutlass.h"
#include "cutlass/bfloat16.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"


namespace py = pybind11;
using namespace cute;

template <typename T>
T* tensor_ptr(py::handle tensor) {
    auto address = tensor.attr("data_ptr")().cast<uintptr_t>();
    return reinterpret_cast<T*>(address);
}

inline void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <int TBM, int TBN, int TBK, int WM, int WN, int WK, int NumStages, int AlignmentAB = 16>
struct GeometryInt8Gemm {
    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementOutput = cutlass::bfloat16_t;
    using ElementAcc = int32_t;
    using ElementCompute = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    static constexpr int AlignA = AlignmentAB;
    static constexpr int AlignB = AlignmentAB;
    static constexpr int AlignC = 8;
    using TB = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
    using Warp = cutlass::gemm::GemmShape<WM, WN, WK>;
    using Inst = cutlass::gemm::GemmShape<16, 8, 32>;
    static constexpr int EVTStages = 1;
    static constexpr int Threads = (TBM / WM) * (TBN / WN) * 32;

    using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
        TB, Warp, ElementOutput, AlignC, EVTStages>;
    using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
    using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<
        ThreadMap, ElementCompute, cute::Stride<_1, _0, int32_t>>;
    using WScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<
        ThreadMap, ElementCompute, cute::Stride<_0, _1, int32_t>>;
    using Mul0 = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::multiplies,
        ElementCompute,
        ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT0 = cutlass::epilogue::threadblock::Sm80EVT<Mul0, Accum, XScale>;
    using Mul1 = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::multiplies,
        ElementOutput,
        ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT1 = cutlass::epilogue::threadblock::Sm80EVT<Mul1, EVT0, WScale>;
    using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<
        ThreadMap,
        ElementOutput,
        cutlass::FloatRoundStyle::round_to_nearest,
        cute::Stride<int64_t, _1, int64_t>>;
    using EVTD = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT1>;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        ElementA,
        LayoutA,
        cutlass::ComplexTransform::kNone,
        AlignA,
        ElementB,
        LayoutB,
        cutlass::ComplexTransform::kNone,
        AlignB,
        ElementOutput,
        LayoutC,
        AlignC,
        ElementAcc,
        ElementCompute,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        TB,
        Warp,
        Inst,
        EVTD,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        NumStages,
        cutlass::arch::OpMultiplyAddSaturate,
        EVTStages>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    static constexpr size_t shared_storage_bytes() {
        return sizeof(typename GemmKernel::SharedStorage);
    }

    static bool run(
        const int8_t* a,
        const int8_t* b,
        const float* x_scale,
        const float* weight_scale,
        ElementOutput* output,
        int m,
        int n,
        int k,
        cudaStream_t stream) {
        cutlass::gemm::GemmCoord problem(m, n, k);
        typename EVTD::Arguments callbacks{
            {
                {
                    {},
                    {const_cast<float*>(x_scale), 0.f, {_1{}, _0{}, m}},
                    {},
                },
                {const_cast<float*>(weight_scale), 0.f, {_0{}, _1{}, n}},
                {},
            },
            {output, {n, _1{}, static_cast<int64_t>(m) * n}},
        };
        typename Gemm::Arguments args(
            cutlass::gemm::GemmUniversalMode::kGemm,
            problem,
            1,
            callbacks,
            const_cast<int8_t*>(a),
            const_cast<int8_t*>(b),
            nullptr,
            nullptr,
            static_cast<int64_t>(m) * k,
            static_cast<int64_t>(n) * k,
            0,
            0,
            k,
            k,
            0,
            0);

        Gemm gemm;
        if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
            return false;
        }
        if (Gemm::get_workspace_size(args) != 0) {
            throw std::runtime_error("geometry GEMM unexpectedly requested workspace");
        }
        if (gemm.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) {
            return false;
        }
        return gemm(stream) == cutlass::Status::kSuccess;
    }

    static std::vector<int64_t> attributes() {
        constexpr size_t dynamic_shared = shared_storage_bytes();
        if (dynamic_shared >= 48 * 1024) {
            check_cuda(
                cudaFuncSetAttribute(
                    cutlass::Kernel<GemmKernel>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(dynamic_shared)),
                "cudaFuncSetAttribute");
        }
        cudaFuncAttributes attributes{};
        check_cuda(
            cudaFuncGetAttributes(&attributes, cutlass::Kernel<GemmKernel>),
            "cudaFuncGetAttributes");
        int active_ctas = 0;
        check_cuda(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_ctas,
                cutlass::Kernel<GemmKernel>,
                Threads,
                dynamic_shared),
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
        return {
            attributes.numRegs,
            static_cast<int64_t>(attributes.sharedSizeBytes),
            attributes.maxDynamicSharedSizeBytes,
            static_cast<int64_t>(attributes.localSizeBytes),
            attributes.maxThreadsPerBlock,
            attributes.binaryVersion,
            attributes.ptxVersion,
            static_cast<int64_t>(dynamic_shared),
            active_ctas,
        };
    }
};

using Config0 = GeometryInt8Gemm<128, 256, 64, 64, 64, 64, 3>;
using Config1 = GeometryInt8Gemm<64, 512, 64, 64, 64, 64, 2>;
using Config2 = GeometryInt8Gemm<64, 512, 64, 64, 64, 64, 3>;
using Config3 = GeometryInt8Gemm<64, 512, 64, 32, 128, 64, 2>;
using Config5 = GeometryInt8Gemm<128, 512, 64, 64, 128, 64, 2>;

template <typename Config>
bool run_config(
    py::handle x,
    py::handle weight,
    py::handle x_scale,
    py::handle weight_scale,
    py::handle output,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
    return Config::run(
        tensor_ptr<const int8_t>(x),
        tensor_ptr<const int8_t>(weight),
        tensor_ptr<const float>(x_scale),
        tensor_ptr<const float>(weight_scale),
        tensor_ptr<cutlass::bfloat16_t>(output),
        m,
        n,
        k,
        stream);
}

bool run(
    py::handle x,
    py::handle weight,
    py::handle x_scale,
    py::handle weight_scale,
    py::handle output,
    int config,
    uintptr_t stream_pointer) {
    const int m = x.attr("size")(0).cast<int>();
    const int k = x.attr("size")(1).cast<int>();
    const int n = weight.attr("size")(0).cast<int>();
    auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
    switch (config) {
        case 0: return run_config<Config0>(x, weight, x_scale, weight_scale, output, m, n, k, stream);
        case 1: return run_config<Config1>(x, weight, x_scale, weight_scale, output, m, n, k, stream);
        case 2: return run_config<Config2>(x, weight, x_scale, weight_scale, output, m, n, k, stream);
        case 3: return run_config<Config3>(x, weight, x_scale, weight_scale, output, m, n, k, stream);
        case 4: throw std::invalid_argument("32x512 configuration was rejected by CUTLASS at compile time");
        case 5: return run_config<Config5>(x, weight, x_scale, weight_scale, output, m, n, k, stream);
        default: throw std::invalid_argument("unknown CUTLASS geometry configuration");
    }
}

std::vector<int64_t> kernel_attributes(int config) {
    switch (config) {
        case 0: return Config0::attributes();
        case 1: return Config1::attributes();
        case 2: return Config2::attributes();
        case 3: return Config3::attributes();
        case 4: throw std::invalid_argument("32x512 configuration was rejected by CUTLASS at compile time");
        case 5: return Config5::attributes();
        default: throw std::invalid_argument("unknown CUTLASS geometry configuration");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run", &run);
    module.def("kernel_attributes", &kernel_attributes);
}
