// src/core/dflash2_q4_parity.cpp - the drafter's Q4_0 GEMM (kernels/dflash2.hpp df2_gemm with Df2Mat::q4, after df2_q4_tile) against a CPU reference.
//
//     dflash2_q4_parity [--batch-wmma]
//
// Random float matrices are quantized to ggml Q4_0 blocks on the CPU (the reference of tools/dflash2_pack.py quant_q4_0), tiled on the device, and multiplied by
// random planar-int8 activations (df2_quant_rows).  The reference reads the same blocks and the same activation codes back and sums per block in integers,
// so the only difference to the kernel is the float order of the sum over blocks.  Shapes: the drafter's (n_in 2560, 4096, 7680, 12800), one, two and three
// matrices to a launch, 1..8 columns. With --batch-wmma and STRATA_DF2_BATCH_WMMA=1 on gfx1151, also compare
// odd and full column tiles against the legacy split path and the CPU reference.
#include "strata/kernels/dflash2.hpp"
#include "strata/kernels/gfx_arch.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {

uint16_t f2h(float f) {   // IEEE half, round to nearest even (the d of a block is stored this way)
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0) {
        if (e < -10) return (uint16_t) sign;
        m |= 0x800000u;
        const int sh = 14 - e;
        uint32_t r = m >> sh;
        const uint32_t rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1))) ++r;
        return (uint16_t) (sign | r);
    }
    if (e >= 31) return (uint16_t) (sign | 0x7c00u);
    uint32_t r = ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) ++r;
    return (uint16_t) (sign | r);
}

float h2f(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t x;
    if (e == 0) {
        if (m == 0) x = sign;
        else {
            int s = 0;
            uint32_t mm = m;
            while (!(mm & 0x400)) { mm <<= 1; ++s; }
            x = sign | ((uint32_t) (127 - 15 - s + 1) << 23) | ((mm & 0x3ff) << 13);
        }
    } else if (e == 31) x = sign | 0x7f800000u | (m << 13);
    else x = sign | ((e - 15 + 127) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

// ggml quantize_row_q4_0_ref: 18 bytes a block
void quant_q4_0(const float* x, int n, uint8_t* out) {
    for (int b = 0; b < n / 32; ++b) {
        float amax = 0, mx = 0;
        for (int i = 0; i < 32; ++i)
            if (std::fabs(x[b * 32 + i]) > amax) { amax = std::fabs(x[b * 32 + i]); mx = x[b * 32 + i]; }
        const float d = mx / -8.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        const uint16_t dh = f2h(d);
        out[b * 18] = (uint8_t) (dh & 0xff);
        out[b * 18 + 1] = (uint8_t) (dh >> 8);
        for (int i = 0; i < 16; ++i) {
            const int q0 = std::min(15, (int) (x[b * 32 + i] * id + 8.5f)), q1 = std::min(15, (int) (x[b * 32 + 16 + i] * id + 8.5f));
            out[b * 18 + 2 + i] = (uint8_t) (q0 | (q1 << 4));
        }
    }
}

struct Case {
    int n_in;
    std::vector<int> n_out;
    std::vector<int> columns{1, 2, 3, 5, 8};
};

}  // namespace

int main(int argc, char** argv) {
    const bool batch_wmma = argc == 2 && std::strcmp(argv[1], "--batch-wmma") == 0;
    if (argc != 1 && !batch_wmma) {
        std::fprintf(stderr, "usage: dflash2_q4_parity [--batch-wmma]\n");
        return 2;
    }
    if (batch_wmma) {
#if defined(STRATA_USE_HIP)
        const char* flag = std::getenv("STRATA_DF2_BATCH_WMMA");
        cudaDeviceProp prop{};
        int device = 0;
        if (flag == nullptr || std::strcmp(flag, "1") != 0 || cudaGetDevice(&device) != cudaSuccess ||
            cudaGetDeviceProperties(&prop, device) != cudaSuccess || !strata::kernels::gfx_arch_is_gfx1151(prop.gcnArchName)) {
            std::fprintf(stderr, "--batch-wmma requires gfx1151 and STRATA_DF2_BATCH_WMMA=1\n");
            return 2;
        }
#else
        std::fprintf(stderr, "--batch-wmma requires a HIP build\n");
        return 2;
#endif
    }
    std::mt19937 rng(20261031);
    std::normal_distribution<float> nd(0.0f, 0.1f);
    const std::vector<Case> cases = batch_wmma
        ? std::vector<Case>{{2560, {16, 32, 48}, {9, 17, 31, 63, 64, 65}},
                            {2560, {4096, 1024, 1024}, {17}}, {2560, {1024, 1024}, {64, 65}},
                            {12800, {2560}, {17}}, {4096, {2560}, {9}}, {7680, {2560}, {9}}}
        : std::vector<Case>{{2560, {64}}, {2560, {64, 32}}, {2560, {4096, 1024, 1024}}, {4096, {2560}},
                            {7680, {2560}}, {12800, {2560}}, {2560, {7680, 7680}}, {2560, {16}}};
    int bad = 0, total = 0;
    for (const Case& cs : cases) {
        const int n_in = cs.n_in, nb = n_in / 32, nm = (int) cs.n_out.size();
        const int max_cols = *std::max_element(cs.columns.begin(), cs.columns.end());
        const int padded_cols = (max_cols + 7) & ~7;  // The legacy tail launches read a full eight input columns.
        std::vector<std::vector<uint8_t>> blocks(nm);
        std::vector<uint8_t*> dev_w(nm);
        std::vector<float*> dev_y(nm);
        for (int m = 0; m < nm; ++m) {
            const int n_out = cs.n_out[m];
            std::vector<float> w((size_t) n_out * n_in);
            for (float& v : w) v = nd(rng);
            blocks[m].resize((size_t) n_out * nb * 18);
            for (int r = 0; r < n_out; ++r) quant_q4_0(&w[(size_t) r * n_in], n_in, &blocks[m][(size_t) r * nb * 18]);
            uint8_t* tmp;
            cudaMalloc((void**) &tmp, blocks[m].size());
            cudaMemcpy(tmp, blocks[m].data(), blocks[m].size(), cudaMemcpyHostToDevice);
            cudaMalloc((void**) &dev_w[m], blocks[m].size());
            strata::kernels::df2_q4_tile(tmp, dev_w[m], dev_w[m] + (size_t) n_out * n_in / 2, n_in, (int64_t) n_out * nb, nullptr);
            cudaDeviceSynchronize();
            cudaFree(tmp);
            cudaMalloc((void**) &dev_y[m], (size_t) (padded_cols + 1) * n_out * sizeof(float));
        }
        std::vector<float> x((size_t) padded_cols * n_in);
        for (size_t i = 0; i < (size_t) max_cols * n_in; ++i) x[i] = nd(rng) * 4.0f;
        float* dx;
        int8_t* dxq;
        float* dxd;
        cudaMalloc((void**) &dx, x.size() * sizeof(float));
        cudaMalloc((void**) &dxq, x.size());
        cudaMalloc((void**) &dxd, (size_t) padded_cols * nb * sizeof(float));
        cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
        strata::kernels::df2_quant_rows(dx, n_in, padded_cols, dxq, dxd, nullptr);
        cudaDeviceSynchronize();
        std::vector<int8_t> xq(x.size());
        std::vector<float> xd((size_t) padded_cols * nb);
        cudaMemcpy(xq.data(), dxq, xq.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(xd.data(), dxd, xd.size() * sizeof(float), cudaMemcpyDeviceToHost);
        for (int ncols : cs.columns) {
            strata::kernels::Df2Mat mats[3];
            for (int m = 0; m < nm; ++m) mats[m] = strata::kernels::Df2Mat{dev_w[m], dev_y[m], cs.n_out[m], true};
            std::vector<std::vector<float>> legacy(nm);
            if (batch_wmma) {
                // A named kernel variant bypasses the batch WMMA dispatcher. Variant 1
                // uses the same half-block accumulation and lane sum as the other DP4A variants.
                strata::kernels::df2_gemm_variant(1);
                strata::kernels::df2_gemm(mats, nm, dxq, dxd, n_in, ncols, nullptr);
                if (cudaDeviceSynchronize() != cudaSuccess) return 1;
                for (int m = 0; m < nm; ++m) {
                    legacy[m].resize((size_t) ncols * cs.n_out[m]);
                    if (cudaMemcpy(legacy[m].data(), dev_y[m], legacy[m].size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
                        return 1;
                }
                strata::kernels::df2_gemm_variant(0);
            }
            for (int m = 0; m < nm; ++m) {
                if (cudaMemset(dev_y[m], 0xff, (size_t) (padded_cols + 1) * cs.n_out[m] * sizeof(float)) != cudaSuccess)
                    return 1;
            }
            strata::kernels::df2_gemm(mats, nm, dxq, dxd, n_in, ncols, nullptr);
            if (cudaDeviceSynchronize() != cudaSuccess) return 1;
            double worst = 0, worst_legacy = 0, big = 0;
            bool untouched = true;
            for (int m = 0; m < nm; ++m) {
                const int n_out = cs.n_out[m];
                const size_t count = (size_t) ncols * n_out;
                std::vector<float> y((size_t) (ncols + 1) * n_out);
                if (cudaMemcpy(y.data(), dev_y[m], y.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
                    return 1;
                untouched = untouched && std::all_of(y.begin() + count, y.end(), [](float v) { return std::isnan(v); });
                if (batch_wmma) {
                    for (size_t i = 0; i < count; ++i) {
                        const double difference = std::fabs((double) y[i] - legacy[m][i]);
                        worst_legacy = std::max(worst_legacy, std::isfinite(difference) ? difference : std::numeric_limits<double>::infinity());
                        big = std::max(big, std::fabs((double) legacy[m][i]));
                    }
                }
                // Check every element on compact shapes, and a sample from each actual matrix
                // on larger shapes. Legacy comparison above still checks every output there.
                const int stride = batch_wmma ? std::max(1, n_out / 48) : 1;
                for (int c = 0; c < ncols; ++c)
                    for (int r = 0; r < n_out; r += stride) {
                        double ref = 0;
                        for (int b = 0; b < nb; ++b) {
                            const uint8_t* blk = &blocks[m][((size_t) r * nb + b) * 18];
                            const uint16_t dh = (uint16_t) blk[0] | ((uint16_t) blk[1] << 8);
                            long isum = 0;
                            for (int i = 0; i < 16; ++i) {
                                isum += (long) ((blk[2 + i] & 15) - 8) * xq[(size_t) c * n_in + b * 32 + i];
                                isum += (long) ((blk[2 + i] >> 4) - 8) * xq[(size_t) c * n_in + b * 32 + 16 + i];
                            }
                            ref += (double) h2f(dh) * (double) xd[(size_t) c * nb + b] * (double) isum;
                        }
                        const double got = y[(size_t) c * n_out + r];
                        const double difference = std::fabs(got - ref);
                        worst = std::max(worst, std::isfinite(difference) ? difference : std::numeric_limits<double>::infinity());
                        big = std::max(big, std::fabs(ref));
                    }
            }
            const double tolerance = 2e-5 * std::max(big, 1e-9);
            const bool ok = untouched && worst <= tolerance && worst_legacy <= tolerance;
            ++total;
            if (!ok) ++bad;
            std::printf("n_in %5d mats %d rows %5d cols %2d: CPU max error %.3g / %.3g, legacy max error %.3g, tail %s  %s\n",
                        n_in, nm, cs.n_out[0], ncols, worst, big, worst_legacy, untouched ? "untouched" : "written", ok ? "ok" : "MISMATCH");
        }
        for (int m = 0; m < nm; ++m) { cudaFree(dev_w[m]); cudaFree(dev_y[m]); }
        cudaFree(dx); cudaFree(dxq); cudaFree(dxd);
    }
    std::printf("%s: %d of %d cases match the reference\n", bad == 0 ? "PASS" : "FAIL", total - bad, total);
    return bad == 0 ? 0 : 1;
}
