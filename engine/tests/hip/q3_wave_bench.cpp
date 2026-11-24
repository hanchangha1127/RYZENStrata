// Synthetic dense Q3_K parity and cold timing on one wave32 gfx1151.
// Run hip_q3_wave_bench [calls per timed batch]. No model files are accessed.
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace strata::kernels {
// Low-level dispatch for correctness coverage of the candidate's tails and unselected widths.
bool rdna_kq_mmvq(int type, const void* w, const void* xq, float* y, int n_in, int n_out, int ncols, void* stream);
}

namespace {

struct Shape {
    int k, rows, count;
};
// Projection counts from the pinned ISTA-DASLab Q2_0 GGUF, excluding its embedding.
constexpr Shape shapes[] = {
    {2560, 512, 4}, {2560, 640, 34}, {2560, 6144, 25}, {2560, 10240, 18},
    {2560, 12288, 6}, {6144, 2560, 3}, {768, 643, 0},
};

struct Q81 {
    uint16_t d, sum;
    int8_t code[32];
};
static_assert(sizeof(Q81) == 36);
constexpr int q3_values = 256, q3_bytes = 110, max_columns = 8;

void require(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    std::exit(1);
}

void arm(bool wave) {
#if defined(_WIN32)
    const int error = _putenv_s("STRATA_RYZEN_Q3_WAVE", wave ? "1" : "0");
#else
    const int error = setenv("STRATA_RYZEN_Q3_WAVE", wave ? "1" : "0", 1);
#endif
    if (error != 0) std::exit(1);
}

float half_float(uint16_t word) {
    const int exponent = (word >> 10) & 31;
    const float m = (word & 1023) / 1024.f;
    const float v = exponent == 0 ? std::ldexp(m, -14) : std::ldexp(1 + m, exponent - 15);
    return word & 0x8000 ? -v : v;
}

uint16_t random_half(std::mt19937& random, bool signed_value) {
    const auto bits = random();
    return (uint16_t) ((signed_value ? ((bits >> 31) << 15) : 0) |
                       ((5u + (bits >> 10) % 5u) << 10) | (bits & 1023u));
}

// Mirrors the pinned ggml-quants.c dequantize_row_q3_K reference, including its
// scale transpose and traversal, independently of the GPU's per-lane indices.
void dequant_q3(const uint8_t* block, float* values) {
    constexpr uint32_t low4 = 0x0f0f0f0f, low2 = 0x03030303;
    uint32_t aux[4];
    uint16_t d;
    std::memcpy(aux, block + 96, 12);
    std::memcpy(&d, block + 108, sizeof(d));
    const uint32_t high = aux[2];
    aux[2] = ((aux[0] >> 4) & low4) | (((high >> 4) & low2) << 4);
    aux[3] = ((aux[1] >> 4) & low4) | (((high >> 6) & low2) << 4);
    aux[0] = (aux[0] & low4) | ((high & low2) << 4);
    aux[1] = (aux[1] & low4) | (((high >> 2) & low2) << 4);
    const auto* scales = reinterpret_cast<const uint8_t*>(aux);
    const auto* qs = block + 32;
    const auto* hi = block;
    unsigned mask = 1;
    int i = 0;
    for (int half = 0; half < 2; ++half) {
        for (int shift = 0; shift < 8; shift += 2, mask <<= 1) {
            for (int group = 0; group < 2; ++group) {
                const float scale = half_float(d) * (scales[i++] - 32);
                for (int lane = 16 * group; lane < 16 * (group + 1); ++lane)
                    *values++ = scale * (((qs[lane] >> shift) & 3) - ((hi[lane] & mask) ? 0 : 4));
            }
        }
        qs += 32;
    }
}

struct Ref { double sum = 0, magnitude = 0; };

Ref reference(const uint8_t* row, const Q81* activation, int nblocks) {
    Ref result;
    for (int b = 0; b < nblocks; ++b) {
        float weights[q3_values];
        dequant_q3(row + (size_t) b * q3_bytes, weights);
        for (int g = 0; g < 8; ++g) {
            const Q81& a = activation[b * 8 + g];
            for (int i = 0; i < 32; ++i) {
                const double term = (double) weights[g * 32 + i] * a.code[i] * half_float(a.d);
                result.sum += term;
                result.magnitude += std::abs(term);
            }
        }
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int calls = argc > 1 ? std::max(16, std::atoi(argv[1])) : 96;
    int device = 0;
    cudaDeviceProp props{};
    require(cudaGetDevice(&device), "get device");
    require(cudaGetDeviceProperties(&props, device), "device properties");
    if (props.warpSize != 32 || std::strncmp(props.gcnArchName, "gfx1151", 7) != 0) return 77;
    cudaStream_t stream;
    cudaEvent_t start, end;
    require(cudaStreamCreate(&stream), "stream");
    require(cudaEventCreate(&start), "start event");
    require(cudaEventCreate(&end), "end event");
    double weighted[2][max_columns] = {};
    double max_legacy_delta = 0;
    for (const Shape shape : shapes) {
        const int nblocks = shape.k / q3_values, acol = shape.k / 32;
        const size_t row_bytes = (size_t) nblocks * q3_bytes;
        std::vector<uint8_t> weights(row_bytes * shape.rows);
        std::mt19937 random((uint32_t) (shape.k * 37 + shape.rows));
        for (auto& byte : weights) byte = (uint8_t) random();
        for (size_t i = 108; i < weights.size(); i += q3_bytes) {
            const uint16_t d = random_half(random, true);
            std::memcpy(weights.data() + i, &d, sizeof(d));
        }
        std::vector<Q81> activations((size_t) acol * max_columns);
        for (auto& a : activations) {
            a.d = random_half(random, false);
            a.sum = 0;  // Q3_K uses the scale and signed codes only.
            for (auto& code : a.code) code = (int8_t) ((int) (random() % 255) - 127);
        }
        const size_t cold_bytes = std::max<size_t>(16u << 20, 4 * (size_t) props.l2CacheSize);
        const size_t copies = shape.count == 0 ? 1 : std::clamp((cold_bytes + weights.size() - 1) / weights.size(), (size_t) 3, (size_t) 32);
        std::vector<void*> dw(copies);
        for (auto& copy : dw) {
            require(cudaMalloc(&copy, weights.size()), "weights");
            require(cudaMemcpy(copy, weights.data(), weights.size(), cudaMemcpyHostToDevice), "weight upload");
        }
        Q81* ax;
        float* dy;
        require(cudaMalloc((void**) &ax, activations.size() * sizeof(Q81)), "activations");
        require(cudaMalloc((void**) &dy, (size_t) shape.rows * max_columns * sizeof(float)), "output");
        require(cudaMemcpy(ax, activations.data(), activations.size() * sizeof(Q81), cudaMemcpyHostToDevice), "activation upload");
        const auto run = [&](int cols, size_t copy) {
            strata::kernels::native_mmvq(11, dw[copy], ax, dy, shape.k, shape.rows, cols, stream);
        };
        const int nref = shape.count == 0 ? shape.rows : 37;
        std::vector<Ref> refs((size_t) nref * max_columns);
        for (int c = 0; c < max_columns; ++c) {
            for (int i = 0; i < nref; ++i) {
                const int row = i * (shape.rows - 1) / (nref - 1);
                refs[(size_t) c * nref + i] = reference(weights.data() + row * row_bytes,
                                                       activations.data() + (size_t) c * acol, nblocks);
            }
        }
        for (int cols = 1; cols <= max_columns; ++cols) {
            std::vector<float> baseline((size_t) shape.rows * cols), candidate(baseline.size());
            double error[2] = {};
            for (int variant : {0, 1}) {
                arm(variant != 0);
                if (variant) {
                    if (!strata::kernels::rdna_kq_mmvq(11, dw[0], ax, dy, shape.k, shape.rows, cols, stream))
                        return 1;
                } else {
                    run(cols, 0);
                }
                require(cudaStreamSynchronize(stream), "parity kernel");
                auto& output = variant ? candidate : baseline;
                require(cudaMemcpy(output.data(), dy, output.size() * sizeof(float), cudaMemcpyDeviceToHost), "parity read");
                for (int c = 0; c < cols; ++c) {
                    for (int i = 0; i < nref; ++i) {
                        const int row = i * (shape.rows - 1) / (nref - 1);
                        const auto ref = refs[(size_t) c * nref + i];
                        const float value = output[(size_t) c * shape.rows + row];
                        const double diff = std::abs(value - ref.sum);
                        error[variant] = std::max(error[variant], diff / std::max(1.0, ref.magnitude));
                        if (!std::isfinite(value) || diff > 3e-6 * ref.magnitude + 2e-6) {
                            std::fprintf(stderr, "CPU agreement failed: variant=%d shape=%dx%d T=%d row=%d column=%d out=%.9g ref=%.9g error=%.9g\n",
                                         variant, shape.k, shape.rows, cols, row, c, value, ref.sum, diff);
                            return 1;
                        }
                    }
                }
            }
            for (size_t i = 0; i < baseline.size(); ++i)
                max_legacy_delta = std::max(max_legacy_delta, std::abs((double) baseline[i] - candidate[i]));
            if (shape.count == 0) continue;  // The odd-row tail is a correctness check.
            double us[2] = {};
            for (int variant : {0, 1, 1, 0}) {
                arm(variant != 0);
                for (int i = 0; i < 20; ++i) run(cols, i % copies);
                require(cudaEventRecord(start, stream), "timing start");
                for (int i = 0; i < calls; ++i) run(cols, i % copies);
                require(cudaEventRecord(end, stream), "timing end");
                require(cudaEventSynchronize(end), "timing wait");
                float ms = 0;
                require(cudaEventElapsedTime(&ms, start, end), "timing elapsed");
                us[variant] += 500.0 * ms / calls;
            }
            for (int v = 0; v < 2; ++v) weighted[v][cols - 1] += shape.count * us[v];
            std::printf("K=%d N=%d count=%d T=%d legacy=%.3f us wave=%.3f us CPU error/max-term-sum=%.3g/%.3g\n",
                        shape.k, shape.rows, shape.count, cols, us[0], us[1], error[0], error[1]);
        }
        for (void* copy : dw) require(cudaFree(copy), "weights free");
        require(cudaFree(ax), "activations free");
        require(cudaFree(dy), "output free");
    }
    for (int c = 0; c < max_columns; ++c)
        std::printf("weighted T=%d legacy=%.3f us wave=%.3f us delta=%.3f us\n",
                    c + 1, weighted[0][c], weighted[1][c], weighted[1][c] - weighted[0][c]);
    std::printf("CPU oracle passed for both routes, including the odd-row tail; largest FP32 legacy difference %.9g\n", max_legacy_delta);
    require(cudaEventDestroy(start), "start destroy");
    require(cudaEventDestroy(end), "end destroy");
    require(cudaStreamDestroy(stream), "stream destroy");
    return max_legacy_delta > 0 ? 0 : 1;  // A fallback in both arms cannot pass.
}
