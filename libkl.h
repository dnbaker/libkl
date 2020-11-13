#ifndef LIBKL_H__
#define LIBKL_H__
#include "x86intrin.h"
#include "sleef.h"
#include <math.h>
#include <unistd.h>
#include <assert.h>


#ifndef LIBKL_API
#define LIBKL_API
#endif


#ifdef __cplusplus
extern "C" {
#endif
// KL divergence
LIBKL_API double kl_reduce_aligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhi, double rhi);
LIBKL_API double kl_reduce_aligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhi, float rhi);
LIBKL_API double kl_reduce_unaligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhi, double rhi);
LIBKL_API double kl_reduce_unaligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhi, float rhi);
// Jensen-Shannon divergence
LIBKL_API double jsd_reduce_aligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhi, double rhi);
LIBKL_API double jsd_reduce_aligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhi, float rhi);
LIBKL_API double jsd_reduce_unaligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhi, double rhi);
LIBKL_API double jsd_reduce_unaligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhi, float rhi);
// Multinomial Log-likelihood ratio test
LIBKL_API double llr_reduce_aligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, size_t n, double lambda, double lhinc, double rhinc);
LIBKL_API double llr_reduce_aligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, size_t n, double lambda, float lhinc, float rhinc);
LIBKL_API double llr_reduce_unaligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, size_t n, double lambda, double lhinc, double rhinc);
LIBKL_API double llr_reduce_unaligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, size_t n, double lambda, float lhinc, float rhinc);
// Itakura-Saito distance
LIBKL_API double is_reduce_aligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhi, double rhi);
LIBKL_API double is_reduce_aligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhi, float rhi);
LIBKL_API double is_reduce_unaligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhi, double rhi);
LIBKL_API double is_reduce_unaligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhi, float rhi);
// Symmetrized Itakura-Saito distance
LIBKL_API double sis_reduce_aligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhi, double rhi);
LIBKL_API double sis_reduce_aligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhi, float rhi);
LIBKL_API double sis_reduce_unaligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhi, double rhi);
LIBKL_API double sis_reduce_unaligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhi, float rhi);
// Bhattacharyya Distance
// Computes BS = sum(sqrt(x) * sqrt(y)), which can be converted into Bhattacharyya metric [sqrt(1 - BS)] or distance [-log(BS)]
LIBKL_API double bhattd_reduce_aligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhmul, double rhmul, double lhi, double rhi);
LIBKL_API double bhattd_reduce_aligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhmul, float rhmul, float lhi, float rhi);
LIBKL_API double bhattd_reduce_unaligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhmul, double rhmul, double lhi, double rhi);
LIBKL_API double bhattd_reduce_unaligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhmul, float rhmul, float lhi, float rhi);
// Hellinger Distance
LIBKL_API double helld_reduce_aligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhmul, double rhmul, double lhi, double rhi);
LIBKL_API double helld_reduce_aligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhmul, float rhmul, float lhi, float rhi);
LIBKL_API double helld_reduce_unaligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhmul, double rhmul, double lhi, double rhi);
LIBKL_API double helld_reduce_unaligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhmul, float rhmul, float lhi, float rhi);

#ifdef __cplusplus
} // extern "C" 

namespace libkl {

static inline double jsd_reduce_unaligned(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhi, double rhi) {
    return jsd_reduce_unaligned_d(lhs, rhs, n, lhi, rhi);
}
static inline double jsd_reduce_unaligned(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhi, float rhi) {
    return jsd_reduce_unaligned_f(lhs, rhs, n, lhi, rhi);
}
static inline double jsd_reduce_aligned(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhi, double rhi) {
    return jsd_reduce_aligned_d(lhs, rhs, n, lhi, rhi);
}
static inline double jsd_reduce_aligned(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhi, float rhi) {
    return jsd_reduce_aligned_f(lhs, rhs, n, lhi, rhi);
}
static inline double sis_reduce_unaligned(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhi, double rhi) {
    return sis_reduce_unaligned_d(lhs, rhs, n, lhi, rhi);
}
static inline double sis_reduce_unaligned(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhi, float rhi) {
    return sis_reduce_unaligned_f(lhs, rhs, n, lhi, rhi);
}
static inline double sis_reduce_aligned(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhi, double rhi) {
    return sis_reduce_aligned_d(lhs, rhs, n, lhi, rhi);
}
static inline double sis_reduce_aligned(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhi, float rhi) {
    return sis_reduce_aligned_f(lhs, rhs, n, lhi, rhi);
}
static inline double is_reduce_unaligned(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhi, double rhi) {
    return is_reduce_unaligned_d(lhs, rhs, n, lhi, rhi);
}
static inline double is_reduce_unaligned(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhi, float rhi) {
    return is_reduce_unaligned_f(lhs, rhs, n, lhi, rhi);
}
static inline double is_reduce_aligned(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhi, double rhi) {
    return is_reduce_aligned_d(lhs, rhs, n, lhi, rhi);
}
static inline double is_reduce_aligned(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhi, float rhi) {
    return is_reduce_aligned_f(lhs, rhs, n, lhi, rhi);
}
static inline double kl_reduce_unaligned(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhi, double rhi) {
    return kl_reduce_unaligned_d(lhs, rhs, n, lhi, rhi);
}
static inline double kl_reduce_unaligned(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhi, float rhi) {
    return kl_reduce_unaligned_f(lhs, rhs, n, lhi, rhi);
}
static inline double kl_reduce_aligned(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhi, double rhi) {
    return kl_reduce_aligned_d(lhs, rhs, n, lhi, rhi);
}
static inline double kl_reduce_aligned(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhi, float rhi) {
    return kl_reduce_aligned_f(lhs, rhs, n, lhi, rhi);
}
static inline double llr_reduce_aligned(const double *const __restrict__ lhs, const double *const __restrict__ rhs, size_t n, double lambda, double lhinc, double rhinc) {
    return llr_reduce_aligned_d(lhs, rhs, n, lambda, lhinc, rhinc);
}
static inline double llr_reduce_aligned(const float *const __restrict__ lhs, const float *const __restrict__ rhs, size_t n, float lambda, float lhinc, float rhinc) {
    return llr_reduce_aligned_f(lhs, rhs, n, lambda, lhinc, rhinc);
}
static inline double llr_reduce_unaligned(const double *const __restrict__ lhs, const double *const __restrict__ rhs, size_t n, double lambda, double lhinc, double rhinc) {
    return llr_reduce_unaligned_d(lhs, rhs, n, lambda, lhinc, rhinc);
}
static inline double llr_reduce_unaligned(const float *const __restrict__ lhs, const float *const __restrict__ rhs, size_t n, float lambda, float lhinc, float rhinc) {
    return llr_reduce_unaligned_f(lhs, rhs, n, lambda, lhinc, rhinc);
}
static inline double bhattd_reduce_aligned(const double *const __restrict__ lhs, const double *const __restrict__ rhs, size_t n,  double lhmul, double rhmul, double lhinc, double rhinc) {
    return bhattd_reduce_aligned_d(lhs, rhs, n, lhmul, rhmul, lhinc, rhinc);
}
static inline double bhattd_reduce_aligned(const float *const __restrict__ lhs, const float *const __restrict__ rhs, size_t n, float lhmul, float rhmul, float lhinc, float rhinc) {
    return bhattd_reduce_aligned_f(lhs, rhs, n, lhmul, rhmul, lhinc, rhinc);
}
static inline double bhattd_reduce_unaligned(const double *const __restrict__ lhs, const double *const __restrict__ rhs, size_t n, double lhmul, double rhmul,  double lhinc, double rhinc) {
    return bhattd_reduce_unaligned_d(lhs, rhs, n, lhmul, rhmul, lhinc, rhinc);
}
static inline double bhattd_reduce_unaligned(const float *const __restrict__ lhs, const float *const __restrict__ rhs, size_t n, float lhmul, float rhmul, float lhinc, float rhinc) {
    return bhattd_reduce_unaligned_f(lhs, rhs, n, lhmul, rhmul, lhinc, rhinc);
}
static inline double helld_reduce_aligned(const double *const __restrict__ lhs, const double *const __restrict__ rhs, size_t n,  double lhmul, double rhmul, double lhinc, double rhinc) {
    return helld_reduce_aligned_d(lhs, rhs, n, lhmul, rhmul, lhinc, rhinc);
}
static inline double helld_reduce_aligned(const float *const __restrict__ lhs, const float *const __restrict__ rhs, size_t n, float lhmul, float rhmul, float lhinc, float rhinc) {
    return helld_reduce_aligned_f(lhs, rhs, n, lhmul, rhmul, lhinc, rhinc);
}
static inline double helld_reduce_unaligned(const double *const __restrict__ lhs, const double *const __restrict__ rhs, size_t n, double lhmul, double rhmul,  double lhinc, double rhinc) {
    return helld_reduce_unaligned_d(lhs, rhs, n, lhmul, rhmul, lhinc, rhinc);
}
static inline double helld_reduce_unaligned(const float *const __restrict__ lhs, const float *const __restrict__ rhs, size_t n, float lhmul, float rhmul, float lhinc, float rhinc) {
    return helld_reduce_unaligned_f(lhs, rhs, n, lhmul, rhmul, lhinc, rhinc);
}
static inline double llr_reduce(const float *const __restrict__ lhs, const float *const __restrict__ rhs, size_t n, float lambda, float lhinc, float rhinc) {
#if __AVX512F__
    if((uint64_t)lhs % 64 == 0 && (uint64_t)rhs % 64 == 0)
#elif __AVX2__
    if((uint64_t)lhs % 32 == 0 && (uint64_t)rhs % 32 == 0)
#elif __SSE2__
    if((uint64_t)lhs % 16 == 0 && (uint64_t)rhs % 16 == 0)
#else
    if(1)
#endif
        return llr_reduce_aligned_f(lhs, rhs, n, lambda, lhinc, rhinc);
    else
        return llr_reduce_unaligned_f(lhs, rhs, n, lambda, lhinc, rhinc);
}
static inline double llr_reduce(const double *const __restrict__ lhs, const double *const __restrict__ rhs, size_t n, double lambda, double lhinc, double rhinc) {
#if __AVX512F__
    if((uint64_t)lhs % 64 == 0 && (uint64_t)rhs % 64 == 0)
#elif __AVX2__
    if((uint64_t)lhs % 32 == 0 && (uint64_t)rhs % 32 == 0)
#elif __SSE2__
    if((uint64_t)lhs % 16 == 0 && (uint64_t)rhs % 16 == 0)
#else
    if(1)
#endif
        return llr_reduce_aligned_d(lhs, rhs, n, lambda, lhinc, rhinc);
    else
        return llr_reduce_unaligned_d(lhs, rhs, n, lambda, lhinc, rhinc);
}
static inline double is_reduce(const float *const __restrict__ lhs, const float *const __restrict__ rhs, size_t n, float lhinc, float rhinc) {
#if __AVX512F__
    if((uint64_t)lhs % 64 == 0 && (uint64_t)rhs % 64 == 0)
#elif __AVX2__
    if((uint64_t)lhs % 32 == 0 && (uint64_t)rhs % 32 == 0)
#elif __SSE2__
    if((uint64_t)lhs % 16 == 0 && (uint64_t)rhs % 16 == 0)
#else
    if(1)
#endif
        return is_reduce_aligned_f(lhs, rhs, n, lhinc, rhinc);
    else
        return is_reduce_unaligned_f(lhs, rhs, n, lhinc, rhinc);
}
static inline double is_reduce(const double *const __restrict__ lhs, const double *const __restrict__ rhs, size_t n, double lhinc, double rhinc) {
#if __AVX512F__
    if((uint64_t)lhs % 64 == 0 && (uint64_t)rhs % 64 == 0)
#elif __AVX2__
    if((uint64_t)lhs % 32 == 0 && (uint64_t)rhs % 32 == 0)
#elif __SSE2__
    if((uint64_t)lhs % 16 == 0 && (uint64_t)rhs % 16 == 0)
#else
    if(1)
#endif
        return is_reduce_aligned_d(lhs, rhs, n, lhinc, rhinc);
    else
        return is_reduce_unaligned_d(lhs, rhs, n, lhinc, rhinc);
}
static inline double sis_reduce(const float *const __restrict__ lhs, const float *const __restrict__ rhs, size_t n, float lhinc, float rhinc) {
#if __AVX512F__
    if((uint64_t)lhs % 64 == 0 && (uint64_t)rhs % 64 == 0)
#elif __AVX2__
    if((uint64_t)lhs % 32 == 0 && (uint64_t)rhs % 32 == 0)
#elif __SSE2__
    if((uint64_t)lhs % 16 == 0 && (uint64_t)rhs % 16 == 0)
#else
    if(1)
#endif
        return sis_reduce_aligned_f(lhs, rhs, n, lhinc, rhinc);
    else
        return sis_reduce_unaligned_f(lhs, rhs, n, lhinc, rhinc);
}
static inline double sis_reduce(const double *const __restrict__ lhs, const double *const __restrict__ rhs, size_t n, double lhinc, double rhinc) {
#if __AVX512F__
    if((uint64_t)lhs % 64 == 0 && (uint64_t)rhs % 64 == 0)
#elif __AVX2__
    if((uint64_t)lhs % 32 == 0 && (uint64_t)rhs % 32 == 0)
#elif __SSE2__
    if((uint64_t)lhs % 16 == 0 && (uint64_t)rhs % 16 == 0)
#else
    if(1)
#endif
        return sis_reduce_aligned_d(lhs, rhs, n, lhinc, rhinc);
    else
        return sis_reduce_unaligned_d(lhs, rhs, n, lhinc, rhinc);
}
static inline double kl_reduce(const float *const __restrict__ lhs, const float *const __restrict__ rhs, size_t n, float lhinc, float rhinc) {
#if __AVX512F__
    if((uint64_t)lhs % 64 == 0 && (uint64_t)rhs % 64 == 0)
#elif __AVX2__
    if((uint64_t)lhs % 32 == 0 && (uint64_t)rhs % 32 == 0)
#elif __SSE2__
    if((uint64_t)lhs % 16 == 0 && (uint64_t)rhs % 16 == 0)
#else
    if(1)
#endif
        return kl_reduce_aligned_f(lhs, rhs, n, lhinc, rhinc);
    else
        return kl_reduce_unaligned_f(lhs, rhs, n, lhinc, rhinc);
}
static inline double kl_reduce(const double *const __restrict__ lhs, const double *const __restrict__ rhs, size_t n, double lhinc, double rhinc) {
#if __AVX512F__
    if((uint64_t)lhs % 64 == 0 && (uint64_t)rhs % 64 == 0)
#elif __AVX2__
    if((uint64_t)lhs % 32 == 0 && (uint64_t)rhs % 32 == 0)
#elif __SSE2__
    if((uint64_t)lhs % 16 == 0 && (uint64_t)rhs % 16 == 0)
#else
    if(1)
#endif
        return kl_reduce_aligned_d(lhs, rhs, n, lhinc, rhinc);
    else
        return kl_reduce_unaligned_d(lhs, rhs, n, lhinc, rhinc);
}
static inline double jsd_reduce(const float *const __restrict__ lhs, const float *const __restrict__ rhs, size_t n, float lhinc, float rhinc) {
#if __AVX512F__
    if((uint64_t)lhs % 64 == 0 && (uint64_t)rhs % 64 == 0)
#elif __AVX2__
    if((uint64_t)lhs % 32 == 0 && (uint64_t)rhs % 32 == 0)
#elif __SSE2__
    if((uint64_t)lhs % 16 == 0 && (uint64_t)rhs % 16 == 0)
#else
    if(1)
#endif
        return jsd_reduce_aligned_f(lhs, rhs, n, lhinc, rhinc);
    else
        return jsd_reduce_unaligned_f(lhs, rhs, n, lhinc, rhinc);
}
static inline double jsd_reduce(const double *const __restrict__ lhs, const double *const __restrict__ rhs, size_t n, double lhinc, double rhinc) {
#if __AVX512F__
    if((uint64_t)lhs % 64 == 0 && (uint64_t)rhs % 64 == 0)
#elif __AVX2__
    if((uint64_t)lhs % 32 == 0 && (uint64_t)rhs % 32 == 0)
#elif __SSE2__
    if((uint64_t)lhs % 16 == 0 && (uint64_t)rhs % 16 == 0)
#else
    if(1)
#endif
        return jsd_reduce_aligned_d(lhs, rhs, n, lhinc, rhinc);
    else
        return jsd_reduce_unaligned_d(lhs, rhs, n, lhinc, rhinc);
}
static inline double bhattd_reduce(const float *const __restrict__ lhs, const float *const __restrict__ rhs, size_t n, float lhmul, float rhmul, float lhinc, float rhinc) {
#if __AVX512F__
    if((uint64_t)lhs % 64 == 0 && (uint64_t)rhs % 64 == 0)
#elif __AVX2__
    if((uint64_t)lhs % 32 == 0 && (uint64_t)rhs % 32 == 0)
#elif __SSE2__
    if((uint64_t)lhs % 16 == 0 && (uint64_t)rhs % 16 == 0)
#else
    if(1)
#endif
        return bhattd_reduce_aligned_f(lhs, rhs, n, lhmul, rhmul, lhinc, rhinc);
    else
        return bhattd_reduce_unaligned_f(lhs, rhs, n, lhmul, rhmul, lhinc, rhinc);
}
static inline double bhattd_reduce(const double *const __restrict__ lhs, const double *const __restrict__ rhs, size_t n, double lhmul, double rhmul, double lhinc, double rhinc) {
#if __AVX512F__
    if((uint64_t)lhs % 64 == 0 && (uint64_t)rhs % 64 == 0)
#elif __AVX2__
    if((uint64_t)lhs % 32 == 0 && (uint64_t)rhs % 32 == 0)
#elif __SSE2__
    if((uint64_t)lhs % 16 == 0 && (uint64_t)rhs % 16 == 0)
#else
    if(1)
#endif
        return bhattd_reduce_aligned_d(lhs, rhs, n, lhmul, rhmul, lhinc, rhinc);
    else
        return bhattd_reduce_unaligned_d(lhs, rhs, n, lhmul, rhmul, lhinc, rhinc);
}
static inline double helld_reduce(const float *const __restrict__ lhs, const float *const __restrict__ rhs, size_t n, float lhmul, float rhmul, float lhinc, float rhinc) {
#if __AVX512F__
    if((uint64_t)lhs % 64 == 0 && (uint64_t)rhs % 64 == 0)
#elif __AVX2__
    if((uint64_t)lhs % 32 == 0 && (uint64_t)rhs % 32 == 0)
#elif __SSE2__
    if((uint64_t)lhs % 16 == 0 && (uint64_t)rhs % 16 == 0)
#else
    if(1)
#endif
        return helld_reduce_aligned_f(lhs, rhs, n, lhmul, rhmul, lhinc, rhinc);
    else
        return helld_reduce_unaligned_f(lhs, rhs, n, lhmul, rhmul, lhinc, rhinc);
}
static inline double helld_reduce(const double *const __restrict__ lhs, const double *const __restrict__ rhs, size_t n, double lhmul, double rhmul, double lhinc, double rhinc) {
#if __AVX512F__
    if((uint64_t)lhs % 64 == 0 && (uint64_t)rhs % 64 == 0)
#elif __AVX2__
    if((uint64_t)lhs % 32 == 0 && (uint64_t)rhs % 32 == 0)
#elif __SSE2__
    if((uint64_t)lhs % 16 == 0 && (uint64_t)rhs % 16 == 0)
#else
    if(1)
#endif
        return helld_reduce_aligned_d(lhs, rhs, n, lhmul, rhmul, lhinc, rhinc);
    else
        return helld_reduce_unaligned_d(lhs, rhs, n, lhmul, rhmul, lhinc, rhinc);
}

} // namespace libkl

#endif // #ifdef __cplusplus

#endif /* #ifndef LIBKL_H__ */
