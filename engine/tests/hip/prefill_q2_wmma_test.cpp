// Compare the gfx1151 fusion's grouped gate/up and down against a host reference.
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include "strata/prefill/moe_fused_iq.hpp"

#include <hip/hip_runtime.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
namespace fused = strata::prefill::fused;
using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;

constexpr int kHidden = fused::kMoeEmbd;
constexpr int kInner = fused::kMoeFF;
constexpr int kQ2 = 42;
constexpr size_t kGateRow = kHidden / 64 * 18;
constexpr size_t kDownRow = kInner / 64 * 18;
constexpr size_t kUp = kGateRow * kInner;
constexpr size_t kDown = 2 * kUp;
constexpr size_t kBlob = kDown + kDownRow * kHidden;

void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}

struct Buffer {
    void* ptr = nullptr;
    explicit Buffer(size_t bytes) { check(hipMalloc(&ptr, bytes)); }
    ~Buffer() { (void) hipFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    template <class T> T* data() const { return static_cast<T*>(ptr); }
    void upload(const void* source, size_t bytes) { check(hipMemcpy(ptr, source, bytes, hipMemcpyHostToDevice)); }
};

template <int Width>
double dot(const uint8_t* row, const uint16_t* input) {
    double sum = 0;
    for (int column = 0; column < Width; column += 64) {
        float weights[64];
        strata::dequantize_q2_0(row + (size_t) (column / 64) * 18, weights);
        for (int j = 0; j < 64; ++j) sum += (double) weights[j] * f32_from_f16(input[column + j]);
    }
    return sum;
}
}  // namespace

int main() try {
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess || !strata::kernels::gfx_arch_is_gfx1151(prop.gcnArchName)) return 77;
#if defined(_WIN32)
    _putenv_s("STRATA_PF_FUSED", "1");
    _putenv_s("STRATA_PF_F16", "2");
    _putenv_s("STRATA_RYZEN_Q2_WMMA", "1");
#else
    setenv("STRATA_PF_FUSED", "1", 1);
    setenv("STRATA_PF_F16", "2", 1);
    setenv("STRATA_RYZEN_Q2_WMMA", "1", 1);
#endif
    if (!fused::native_supported(kQ2, kQ2)) throw std::runtime_error("Q2_0 fusion was not admitted");

    constexpr std::array counts{0, 1, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 127, 128, 129};
    constexpr int kExperts = static_cast<int>(counts.size());
    std::vector<int32_t> ids;
    std::vector<int> samples;
    for (int e = 0; e < kExperts; ++e) {
        const int first = static_cast<int>(ids.size());
        if (counts[e] > 0) samples.push_back(first);
        if (counts[e] > 1) samples.push_back(first + counts[e] - 1);
        ids.insert(ids.end(), counts[e], e);
    }
    const size_t rows = ids.size();
    std::mt19937 rng(395);
    std::vector<uint8_t> blobs(kExperts * kBlob);
    for (size_t off = 0; off < blobs.size(); off += 18) {
        const uint16_t scale = f16_from_f32((1.0f + (float) (rng() % 7)) / 2048.0f);
        std::memcpy(blobs.data() + off, &scale, 2);
        for (size_t b = 2; b < 18; ++b) blobs[off + b] = static_cast<uint8_t>(rng());
    }
    std::vector<uint16_t> input(rows * kHidden);
    for (uint16_t& v : input) v = f16_from_f32((static_cast<int>(rng() % 33) - 16) / 32.0f);
    Buffer weights(blobs.size()), x(input.size() * 2), scratch(fused::group_bytes(rows, kExperts));
    Buffer device_ids(rows * 4), slots(rows * 4), source(rows * 4), hidden(rows * kInner * 2), down(rows * kHidden * 4);
    weights.upload(blobs.data(), blobs.size());
    x.upload(input.data(), input.size() * 2);
    device_ids.upload(ids.data(), rows * 4);
    fused::group(device_ids.data<int32_t>(), rows, 1, kExperts, scratch.ptr, slots.data<int32_t>(), source.data<int32_t>(), nullptr);
    const fused::NativeGeom geometry{kQ2, kQ2, kGateRow, kDownRow, kUp, kDown};
    fused::p1a_input(x.ptr, rows * kInner * 2);

    for (bool half_output : {false, true}) {
        check(hipMemset(down.ptr, 0xff, rows * kHidden * 4));
        fused::want_dm_f16(half_output);
        for (int first = 0; first < kExperts; first += 3) {
            fused::Batch batch;
            batch.e0 = first;
            batch.e1 = (first + 3 < kExperts) ? first + 3 : kExperts;
            for (int e = batch.e0; e < batch.e1; ++e)
                batch.blob[e - batch.e0] = weights.data<uint8_t>() + (size_t) e * kBlob;
            fused::experts_native(batch, geometry, kExperts, rows, scratch.ptr, nullptr, source.data<int32_t>(),
                                  hidden.ptr, down.data<float>(), nullptr);
        }
        check(hipDeviceSynchronize());
        if (fused::dm_f16_used() != half_output) throw std::runtime_error("unexpected down output layout");
        std::vector<int32_t> row_map(rows);
        std::vector<float> output(rows * kHidden);
        check(hipMemcpy(row_map.data(), slots.ptr, rows * 4, hipMemcpyDeviceToHost));
        if (half_output) {
            std::vector<uint16_t> packed(output.size());
            check(hipMemcpy(packed.data(), down.ptr, packed.size() * 2, hipMemcpyDeviceToHost));
            for (size_t i = 0; i < output.size(); ++i) output[i] = f32_from_f16(packed[i]);
        } else {
            check(hipMemcpy(output.data(), down.ptr, output.size() * 4, hipMemcpyDeviceToHost));
        }
        for (float value : output) if (!std::isfinite(value)) throw std::runtime_error("missing or non-finite down row");

        double squared_error = 0, squared_reference = 0;
        for (int token : samples) {
            const uint8_t* blob = blobs.data() + (size_t) ids[token] * kBlob;
            const uint16_t* act = input.data() + (size_t) token * kHidden;
            uint16_t intermediate[kInner];
            for (int j = 0; j < kInner; ++j) {
                const double gate = dot<kHidden>(blob + j * kGateRow, act);
                const double up = dot<kHidden>(blob + kUp + j * kGateRow, act);
                intermediate[j] = f16_from_f32((float) (gate * up / (1 + std::exp(-gate))));
            }
            const size_t row = static_cast<size_t>(row_map[token]);
            if (row >= rows) throw std::runtime_error("invalid routing row");
            for (int j = 0; j < kHidden; ++j) {
                const double expected = dot<kInner>(blob + kDown + (size_t) j * kDownRow, intermediate);
                const double error = output[row * kHidden + j] - expected;
                squared_error += error * error;
                squared_reference += expected * expected;
            }
        }
        const double rms = std::sqrt(squared_error / squared_reference);
        std::printf("gfx1151 Q2_0 WMMA: %s down, %zu sampled rows, relative RMS %.6g\n",
                    half_output ? "FP16" : "FP32", samples.size(), rms);
        if (!(rms < 0.003)) return 1;
    }
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "gfx1151 Q2_0 WMMA test: %s\n", error.what());
    return 1;
}
