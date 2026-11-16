// GWC matching deployed eager PyTorch: normalize in FP32, round each operand
// and product to FP16, accumulate in FP32, then round the sum to FP16.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

__global__ void gwc(const __half* left, const __half* right, __half* out,
                    int B, int C, int H, int W, int D, int G, bool normalize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int count = B * G * D * H * W;
    if (i >= count) return;
    int x = i % W, y = (i / W) % H, d = (i / (W * H)) % D;
    int g = (i / (W * H * D)) % G, b = i / (W * H * D * G);
    if (x < d) { out[i] = __float2half(0.f); return; }
    int K = C / G, stride = H * W;
    int li = (b * C + g * K) * stride + y * W + x;
    int ri = (b * C + g * K) * stride + y * W + x - d;
    float ln = 0.f, rn = 0.f;
    if (normalize) {
        for (int k = 0; k < K; ++k) {
            float l = __half2float(left[li + k * stride]);
            float r = __half2float(right[ri + k * stride]);
            ln += l * l; rn += r * r;
        }
        ln = fmaxf(sqrtf(ln), 1e-12f); rn = fmaxf(sqrtf(rn), 1e-12f);
    }
    float sum = 0.f;
    for (int k = 0; k < K; ++k) {
        __half l = left[li + k * stride], r = right[ri + k * stride];
        if (normalize) {
            l = __float2half(__half2float(l) / ln);
            r = __float2half(__half2float(r) / rn);
        }
        sum += __half2float(__hmul(l, r));
    }
    out[i] = __float2half(sum);
}

extern "C" int ffs_gwc(const void* left, const void* right, void* out,
                        int B, int C, int H, int W, int D, int G,
                        bool normalize, void* stream) {
    if (B <= 0 || C <= 0 || H <= 0 || W <= 0 || D <= 0 || G <= 0 || C % G)
        return int(cudaErrorInvalidValue);
    int count = B * G * D * H * W;
    gwc<<<(count + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __half*>(left), static_cast<const __half*>(right),
        static_cast<__half*>(out), B, C, H, W, D, G, normalize);
    return int(cudaGetLastError());
}
