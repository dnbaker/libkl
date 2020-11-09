#include "x86intrin.h"
#include "sleef.h"
#include <math.h>
#include <unistd.h>
#include <assert.h>


#ifdef __cplusplus
extern "C" {
#endif
double __kl_reduce_aligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhi, double rhi);
double __kl_reduce_aligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhi, float rhi);
double __llr_reduce_aligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, size_t n, double lambda, double lhinc, double rhinc);
double __llr_reduce_aligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, size_t n, double lambda, float lhinc, float rhinc);
double __kl_reduce_unaligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhi, double rhi);
double __kl_reduce_unaligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhi, float rhi);
double __llr_reduce_unaligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, size_t n, double lambda, double lhinc, double rhinc);
double __llr_reduce_unaligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, size_t n, double lambda, float lhinc, float rhinc);

#ifdef __cplusplus
} // extern "C" 
double __kl_reduce_aligned(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhi, double rhi) {
    return __kl_reduce_aligned_d(lhs, rhs, n, lhi, rhi);
}
double __kl_reduce_aligned(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhi, float rhi) {
    return __kl_reduce_aligned_f(lhs, rhs, n, lhi, rhi);
}
double __llr_reduce_aligned(const double *const __restrict__ lhs, const double *const __restrict__ rhs, size_t n, double lambda, double lhinc, double rhinc) {
    return __llr_reduce_aligned_d(lhs, rhs, n, lambda, lhinc, rhinc);
}
double __llr_reduce_aligned(const float *const __restrict__ lhs, const float *const __restrict__ rhs, size_t n, float lambda, float lhinc, float rhinc) {
    return __llr_reduce_aligned_f(lhs, rhs, n, lambda, lhinc, rhinc);
}
double __kl_reduce_unaligned(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhi, double rhi) {
    return __kl_reduce_unaligned_d(lhs, rhs, n, lhi, rhi);
}
double __kl_reduce_unaligned(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhi, float rhi) {
    return __kl_reduce_unaligned_f(lhs, rhs, n, lhi, rhi);
}
double __llr_reduce_unaligned(const double *const __restrict__ lhs, const double *const __restrict__ rhs, size_t n, double lambda, double lhinc, double rhinc) {
    return __llr_reduce_unaligned_d(lhs, rhs, n, lambda, lhinc, rhinc);
}
double __llr_reduce_unaligned(const float *const __restrict__ lhs, const float *const __restrict__ rhs, size_t n, float lambda, float lhinc, float rhinc) {
    return __llr_reduce_unaligned_f(lhs, rhs, n, lambda, lhinc, rhinc);
}
double __llr_reduce(const float *const __restrict__ lhs, const float *const __restrict__ rhs, size_t n, float lambda, float lhinc, float rhinc) {
#if __AVX512F__
    if((uint64_t)lhs % 64 == 0 && (uint64_t)rhs % 64 == 0)
#elif __AVX2__
    if((uint64_t)lhs % 32 == 0 && (uint64_t)rhs % 32 == 0)
#elif __SSE2__
    if((uint64_t)lhs % 16 == 0 && (uint64_t)rhs % 16 == 0)
#else
    if(1)
#endif
        return __llr_reduce_aligned_f(lhs, rhs, n, lambda, lhinc, rhinc);
    else
        return __llr_reduce_unaligned_f(lhs, rhs, n, lambda, lhinc, rhinc);
}
double __llr_reduce(const double *const __restrict__ lhs, const double *const __restrict__ rhs, size_t n, double lambda, double lhinc, double rhinc) {
#if __AVX512F__
    if((uint64_t)lhs % 64 == 0 && (uint64_t)rhs % 64 == 0)
#elif __AVX2__
    if((uint64_t)lhs % 32 == 0 && (uint64_t)rhs % 32 == 0)
#elif __SSE2__
    if((uint64_t)lhs % 16 == 0 && (uint64_t)rhs % 16 == 0)
#else
    if(1)
#endif
        return __llr_reduce_aligned_d(lhs, rhs, n, lambda, lhinc, rhinc);
    else
        return __llr_reduce_unaligned_d(lhs, rhs, n, lambda, lhinc, rhinc);
}
double __kl_reduce(const float *const __restrict__ lhs, const float *const __restrict__ rhs, size_t n, float lhinc, float rhinc) {
#if __AVX512F__
    if((uint64_t)lhs % 64 == 0 && (uint64_t)rhs % 64 == 0)
#elif __AVX2__
    if((uint64_t)lhs % 32 == 0 && (uint64_t)rhs % 32 == 0)
#elif __SSE2__
    if((uint64_t)lhs % 16 == 0 && (uint64_t)rhs % 16 == 0)
#else
    if(1)
#endif
        return __kl_reduce_aligned_f(lhs, rhs, n, lhinc, rhinc);
    else
        return __kl_reduce_unaligned_f(lhs, rhs, n, lhinc, rhinc);
}
double __kl_reduce(const double *const __restrict__ lhs, const double *const __restrict__ rhs, size_t n, double lhinc, double rhinc) {
#if __AVX512F__
    if((uint64_t)lhs % 64 == 0 && (uint64_t)rhs % 64 == 0)
#elif __AVX2__
    if((uint64_t)lhs % 32 == 0 && (uint64_t)rhs % 32 == 0)
#elif __SSE2__
    if((uint64_t)lhs % 16 == 0 && (uint64_t)rhs % 16 == 0)
#else
    if(1)
#endif
        return __kl_reduce_aligned_d(lhs, rhs, n, lhinc, rhinc);
    else
        return __kl_reduce_unaligned_d(lhs, rhs, n, lhinc, rhinc);
}

#endif
