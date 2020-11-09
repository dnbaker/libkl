#include "x86intrin.h"
#include "sleef.h"
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include "kl_kernels.h"



// LLR has 3 versions:
// default, DIV1, and LOG3
// LOG3 may have better numerical performance, while default will be faster
// DIV1 seesm to be slower than default and is not recommended.

static inline __attribute__((always_inline)) __m256 broadcast_reduce_add_si256_ps(__m256 x) {
    const __m256 permHalves = _mm256_permute2f128_ps(x, x, 1);
    const __m256 m0 = _mm256_add_ps(permHalves, x);
    const __m256 perm0 = _mm256_permute_ps(m0, 0b01001110);
    const __m256 m1 = _mm256_add_ps(m0, perm0);
    const __m256 perm1 = _mm256_permute_ps(m1, 0b10110001);
    const __m256 m2 = _mm256_add_ps(perm1, m1);
    return m2;
}

static inline __attribute__((always_inline)) __m128 broadcast_reduce_add_si128_ps(__m128 x) {
    __m128 m1 = _mm_shuffle_ps(x, x, _MM_SHUFFLE(0,0,3,2));
    __m128 m2 = _mm_add_ps(x, m1);
    __m128 m3 = _mm_shuffle_ps(m2, m2, _MM_SHUFFLE(0,0,0,1));
    return _mm_add_ps(m2, m3);
}
static inline __attribute__((always_inline)) double hsum_double_avx(__m256d v) {
    __m128d vlow  = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1); // high 128
            vlow  = _mm_add_pd(vlow, vhigh);     // reduce down to 128

    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return  _mm_cvtsd_f64(_mm_add_sd(vlow, high64));  // reduce to scalar
}

double __kl_reduce_aligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhi, double rhi) {
    double ret = 0.;
    size_t i = 0;
#if __AVX512F__
#define lhload(i) _mm512_add_pd(_mm512_load_pd(lhs + ((i) * nper)), lhv)
#define rhload(i) _mm512_add_pd(_mm512_load_pd(rhs + ((i) * nper)), rhv)
    __m512d lhv = _mm512_set1_pd(lhi), rhv = _mm512_set1_pd(rhi);
    assert((uint64_t *)(lhs) % 64 == 0);
    const size_t nper = sizeof(__m512) / sizeof(double);
    const size_t nsimd = n / nper;
    const size_t nsimd4 = (nsimd / 4) * 4;
    {
        __m512d sum = _mm512_setzero_pd();
        for(; i < nsimd4; i += 4) {
            __m512d lh0 = lhload(i), lh1 = lhload(i + 1), lh2 = lhload(i + 2); lh3 = lhload(i + 3);
            __m512d rh0 = rhload(i), rh1 = rhload(i + 1), rh2 = rhload(i + 2); rh3 = rhload(i + 3);
            __m512d v0 = _mm512_mul_pd(lh0, Sleef_logd8_u35(_mm512_div_pd(lh0, rh0)));
            __m512d v1 = _mm512_mul_pd(lh1, Sleef_logd8_u35(_mm512_div_pd(lh1, rh1)));
            __m512d v2 = _mm512_mul_pd(lh2, Sleef_logd8_u35(_mm512_div_pd(lh2, rh2)));
            __m512d v3 = _mm512_mul_pd(lh3, Sleef_logd8_u35(_mm512_div_pd(lh3, rh3)));
            sum = _mm512_add_pd(sum,  _mm512_add_pd(_mm512_add_pd(v0, v1), _mm512_add_pd(v2, v3)));
        }
        ret += _mm512_reduce_add_pd(sum);
    }
    for(; i < nsimd; ++i) {
        __m512d lh = lhload(i), rh = rhload(i);
#undef lhload
#undef rhload
        __m512d logv = Sleef_logd8_u35(_mm512_div_pd(lh, rhload(i)));
        ret += _mm512_reduce_add_pd(_mm512_mul_pd(lh, logv));
    }
    i *= nper;
#elif __AVX2__
    assert(((uint64_t)lhs) % 32 == 0);
    const size_t nper = sizeof(__m256) / sizeof(double);
    const size_t nsimd = n / nper;
    const size_t nsimd4 = (nsimd / 4) * 4;
#define lhload(i) _mm256_add_pd(_mm256_load_pd(lhs + ((i) * nper)), lhv)
#define rhload(i) _mm256_add_pd(_mm256_load_pd(rhs + ((i) * nper)), rhv)
    __m256d lhv = _mm256_set1_pd(lhi), rhv = _mm256_set1_pd(rhi);
    __m256d sum = _mm256_setzero_pd();
    for(; i < nsimd4; i += 4) {
        __m256d lh0 = lhload(i), lh1 = lhload(i + 1), lh2 = lhload(i + 2), lh3 = lhload(i + 3);
        __m256d rh0 = rhload(i), rh1 = rhload(i + 1), rh2 = rhload(i + 2), rh3 = rhload(i + 3);
        __m256d v0 = _mm256_mul_pd(lh0, Sleef_logd4_u35(_mm256_div_pd(lh0, rh0)));
        __m256d v1 = _mm256_mul_pd(lh1, Sleef_logd4_u35(_mm256_div_pd(lh1, rh1)));
        __m256d v2 = _mm256_mul_pd(lh2, Sleef_logd4_u35(_mm256_div_pd(lh2, rh2)));
        __m256d v3 = _mm256_mul_pd(lh3, Sleef_logd4_u35(_mm256_div_pd(lh3, rh3)));
        sum = _mm256_add_pd(sum,  _mm256_add_pd(_mm256_add_pd(v0, v1), _mm256_add_pd(v2, v3)));
    }
    for(; i < nsimd; ++i) {
        __m256d lh = lhload(i), rh = rhload(i);
#undef rhload
#undef lhload
        __m256d res = _mm256_mul_pd(lh, Sleef_logd4_u35(_mm256_div_pd(lh, rh)));
        sum = _mm256_add_pd(sum, res);
    }
    ret = hsum_double_avx(sum);
    i *= nper;
#elif __SSE2__
    assert(((uint64_t)lhs) % 16 == 0);
    const size_t nper = sizeof(__m128) / sizeof(double);
    const size_t nsimd = n / nper;
    __m128d lhv = _mm_set1_pd(lhi), rhv = _mm_set1_pd(rhi);
#define lhload(i) _mm_add_pd(_mm_load_pd(lhs + ((i) * nper)), lhv)
#define rhload(i) _mm_add_pd(_mm_load_pd(rhs + ((i) * nper)), rhv)
    {
        __m128d v = _mm_setzero_pd();
        #pragma GCC unroll 4
        for(; i < nsimd; ++i) {
            __m128d lh = lhload(i), rh = rhload(i);
#undef lhload
#undef rhload
            v = _mm_add_pd(v, _mm_mul_pd(lh, Sleef_logd2_u35(_mm_div_pd(lh, rh))));
        }
        i *= nper;
        ret = v[0] + v[1];
    }
#endif
    #pragma GCC unroll 8
    for(; i < n; ++i) {
        float lhv = lhs[i] + lhi, rhv = rhs[i] + rhi;
        ret += lhv * logf(lhv / rhv);
    }
#ifndef NDEBUG
    double oret = 0.;
    for(size_t j = 0; j < n; ++j) {
        float lhv = lhs[i] + lhi, rhv = rhs[i] + rhi;
        oret += lhv * logf(lhv / rhv);
    }
    assert(fabs(oret - ret) < 1e-5);
#endif
    return ret;
}

double __kl_reduce_aligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhi, float rhi) {
    double ret = 0.;
    size_t i = 0;
#if __AVX512F__
    assert(((uint64_t)lhs) % 64 == 0);
    const size_t nper = sizeof(__m512) / sizeof(float);
    const size_t nsimd = n / nper;
    const size_t nsimd4 = (nsimd / 4) * 4;
    __m512 sum = _mm512_setzero_ps();
    #pragma GCC unroll 4
    for(; i < nsimd; ++i) {
        __m512 lh = _mm512_add_ps(_mm512_load_ps(lhs + (i * nper)), _mm512_set1_ps(lhi));
        __m512 rh = _mm512_add_ps(_mm512_load_ps(rhs + (i * nper)), _mm512_set1_ps(rhi));
        __m512 div = _mm512_div_ps(lh, rh);
        __m512 logv = Sleef_logf16_u35(div);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(lh, logv));
    }
    ret = _mm512_reduce_add_ps(sum);
    i *= nper;
#elif __AVX2__
    assert(((uint64_t)lhs) % 32 == 0);
    const size_t nper = sizeof(__m256) / sizeof(float);
    const size_t nsimd = n / nper;

    __m256 sum = _mm256_setzero_ps();
    #pragma GCC unroll 4
    for(; i < nsimd; ++i) {
        __m256 lh = _mm256_add_ps(_mm256_load_ps(lhs + (i * nper)), _mm256_set1_ps(lhi));
        __m256 rh = _mm256_add_ps(_mm256_load_ps(rhs + (i * nper)), _mm256_set1_ps(rhi));
        __m256 res = _mm256_mul_ps(lh, Sleef_logf8_u35(_mm256_div_ps(lh, rh)));
        sum = _mm256_add_ps(sum, res);
    }
    ret += broadcast_reduce_add_si256_ps(sum)[0];
    i *= nper;
#elif __SSE2__
    assert(reinterpret_cast<uint64_t>(lhs) % 16 == 0);
    const size_t nper = sizeof(__m128) / sizeof(float);
    const size_t nsimd = n / nper;
    __m128 v = _mm_setzero_ps();
    #pragma GCC unroll 4
    for(; i < nsimd; ++i) {
        __m128 lh = _mm_add_ps(_mm_load_ps(lhs + (i * nper)), _mm_set1_ps(lhi));
        __m128 rh = _mm_add_ps(_mm_load_ps(rhs + (i * nper)), _mm_set1_ps(rhi));
        __m128 div = _mm_div_ps(lh, rh);
        __m128 logv = Sleef_logf4_u35(div);
        __m128 res = _mm_mul_ps(lh, logv);
        v = _mm_add_ps(v, res);
    }
    ret += v[0]; ret += v[1]; ret += v[2]; ret += v[3];
    i *= nper;
#endif
    #pragma GCC unroll 8
    for(; i < n; ++i) {
        float lhv = lhs[i] + lhi;
        float rhv = rhs[i] + rhi;
        ret += lhv * logf(lhv / rhv);
    }
#ifndef NDEBUG
    double oret = 0.;
    for(size_t j = 0; j < n; ++j) {
        float lhv = lhs[i] + lhi;
        float rhv = rhs[i] + rhi;
        oret += lhv * logf(lhv / rhv);
    }
    assert(fabs(oret - ret) < 1e-5);
#endif
    return ret;
}

double __llr_reduce_aligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, size_t n, double lambda, float lhinc, float rhinc) {
    const double m1l = 1. - lambda;
    double ret = 0.;
    size_t i = 0;
#if __AVX512F__
    const size_t nperel = sizeof(__m512) / sizeof(float);
    __m512 lhsum = _mm512_setzero_ps(), rhsum = _mm512_setzero_ps();
    size_t nsimd = n / nperel;
    //
    __m512 vlambda = _mm512_set1_ps(lambda), vm1l = _mm512_set1_ps(m1l);
    __m512 lhv = _mm512_set1_ps(lhinc);
    __m512 rhv = _mm512_set1_ps(rhinc);
    #pragma GCC unroll 4
    for(i = 0; i < nsimd; ++i) {
        __m512 lhl = _mm512_add_ps(_mm512_load_ps(&lhs[i * nperel]), lhv);
        __m512 rhl = _mm512_add_ps(_mm512_load_ps(&rhs[i * nperel]), rhv);
#if DIV1
        __m512 mv0 = _mm512_div_ps(_mm512_set1_ps(1.), _mm512_add_ps(_mm512_mul_ps(vlambda, lhl), _mm512_mul_ps(vm1l, rhl)));
        __m512 lv0 = _mm512_mul_ps(lhl, Sleef_logf16_u35(_mm512_mul_ps(lhl, mv)));
        __m512 rv0 = _mm512_mul_ps(rhl, Sleef_logf16_u35(_mm512_mul_ps(rhl, mv)));
#elif LOG3
        __m512 lmv0 = Sleef_logf16_u35(_mm512_add_ps(_mm512_mul_ps(vlambda, lhl), _mm512_mul_ps(vm1l, rhl)));
        __m512 lv0 = _mm512_mul_ps(lhl, _mm512_sub_ps(Sleef_logf16_u35(lhl), lmv0));
        __m512 rv0 = _mm512_mul_ps(rhl, _mm512_sub_ps(Sleef_logf16_u35(rhl), rmv0));
#else
        __m512 mv0 = _mm512_add_ps(_mm512_mul_ps(vlambda, lhl), _mm512_mul_ps(vm1l, rhl));
        __m512 lv0 = _mm512_mul_ps(lhl, Sleef_logf16_u35(_mm512_div_ps(lhl, mv)));
        __m512 rv0 = _mm512_mul_ps(rhl, Sleef_logf16_u35(_mm512_div_ps(rhl, mv)));
#endif
        lhsum = _mm512_add_ps(lhsum, lv0);
        rhsum = _mm512_add_ps(rhsum, rv0);
    }
    ret += lambda * _mm512_reduce_add_ps(lhsum) + m1l * _mm512_reduce_add_ps(rhsum);
    for(i *= nperel;i < n; ++i) {
        float xv = lhs[i] + lhinc;
        float yv = rhs[i] + rhinc;
        float mnv = (xv + yv) * .5;
        ret += lambda * xv * logf(xv / mnv) + m1l * rhs[i] * logf(yv / mnv);
    }
#elif __AVX2__
    const size_t nperel = sizeof(__m256) / sizeof(float);
    __m256 lhsum = _mm256_setzero_ps(), rhsum = _mm256_setzero_ps();
    size_t nsimd = n / nperel;
    __m256 lhv = _mm256_set1_ps(lhinc);
    __m256 rhv = _mm256_set1_ps(rhinc);
    __m256 vlambda = _mm256_set1_ps(lambda), vm1l = _mm256_set1_ps(m1l);
    for(; i < nsimd; ++i) {
        __m256 lhsa = _mm256_add_ps(_mm256_load_ps(&lhs[i * nperel]), lhv);
        __m256 rhsa = _mm256_add_ps(_mm256_load_ps(&rhs[i * nperel]), rhv);
#if DIV1
        __m256 mv = _mm256_div_ps(_mm256_set1_ps(1.), _mm256_add_ps(_mm256_mul_ps(vlambda, lhsa), _mm256_mul_ps(vm1l, rhsa)));
        lhsum = _mm256_add_ps(lhsum, _mm256_mul_ps(lhsa, Sleef_logf8_u35(_mm256_mul_ps(lhsa, mv))));
        rhsum = _mm256_add_ps(rhsum, _mm256_mul_ps(rhsa, Sleef_logf8_u35(_mm256_mul_ps(rhsa, mv))));
#elif LOG3
        __m256 lmv = Sleef_logf8_u35(_mm256_add_ps(_mm256_mul_ps(vlambda, lhsa), _mm256_mul_ps(vm1l, rhsa)));
        lhsum = _mm256_add_ps(lhsum, _mm256_mul_ps(lhsa, _mm256_sub_ps(Sleef_logf8_u35(lhsa), lmv)));
        rhsum = _mm256_add_ps(rhsum, _mm256_mul_ps(rhsa, _mm256_sub_ps(Sleef_logf8_u35(rhsa), lmv)));
#else
        __m256 mv = _mm256_add_ps(_mm256_mul_ps(vlambda, lhsa), _mm256_mul_ps(vm1l, rhsa));
        lhsum = _mm256_add_ps(lhsum, _mm256_mul_ps(lhsa, Sleef_logf8_u35(_mm256_div_ps(lhsa, mv))));
        rhsum = _mm256_add_ps(rhsum, _mm256_mul_ps(rhsa, Sleef_logf8_u35(_mm256_mul_ps(rhsa, mv))));
#endif
    }
    ret += lambda * broadcast_reduce_add_si256_ps(lhsum)[0] + m1l * broadcast_reduce_add_si256_ps(rhsum)[0];
    for(i *= nperel;i < n; ++i) {
        float lhv = lhs[i] + lhinc;
        float rhv = rhs[i] + rhinc;
        float miv = lhv * lambda + rhv * m1l;
        ret += lambda * lhv * logf(lhv / miv)
             + m1l * rhv * logf(rhv / miv);
    }
#elif __SSE2__
    const size_t nperel = sizeof(__m128) / sizeof(float);
    __m128 lhsum = _mm_setzero_ps(), rhsum = _mm_setzero_ps();
    size_t nsimd = n / nperel;
    size_t nsimd4 = nsimd / 4;
    //
    __m128 vlambda = _mm_set1_ps(lambda), vm1l = _mm_set1_ps(m1l);
    for(; i < nsimd; ++i) {
        __m128 lhsa = _mm_add_ps(_mm_load_ps(&lhs[i * nperel]), lhinc);
        __m128 rhsa = _mm_add_ps(_mm_load_ps(&rhs[i * nperel]), rhinc);
#if DIV1
        __m128 mv = _mm_div_ps(_mm_set1_ps(1.), _mm_add_ps(_mm_mul_ps(vlambda, lhsa0), _mm_mul_ps(vm1l, rhsa0)));
        lhsum = _mm_add_ps(lhsum, _mm_mul_ps(lhsa, Sleef_logf4_u35(_mm_mul_ps(lhsa, mv))));
        rhsum = _mm_add_ps(rhsum, _mm_mul_ps(rhsa, Sleef_logf4_u35(_mm_mul_ps(rhsa, mv))));
#elif LOG3
        __m128 lmv = Sleef_logf4_u35(_mm_add_ps(_mm_mul_ps(vlambda, lhsa0), _mm_mul_ps(vm1l, rhsa0)));
        lhsum = _mm_add_ps(lhsum, _mm_mul_ps(lhsa, _mm_sub_ps(Sleef_logf4_u35(lhsa, lmv))));
        rhsum = _mm_add_ps(rhsum, _mm_mul_ps(rhsa, _mm_sub_ps(Sleef_logf4_u35(rhsa, lmv))));
#else
        __m128 mv = _mm_add_ps(_mm_mul_ps(vlambda, lhsa0), _mm_mul_ps(vm1l, rhsa0));
        lhsum = _mm_add_ps(lhsum, _mm_mul_ps(lhsa, Sleef_logf4_u35(_mm_div_ps(lhsa, mv))));
        rhsum = _mm_add_ps(rhsum, _mm_mul_ps(rhsa, Sleef_logf4_u35(_mm_div_ps(rhsa, mv))));
#endif
    }
    ret += lambda * broadcast_reduce_add_si128_ps(lhsum) + m1l * broadcast_reduce_add_si128_ps(rhsum);
    for(i *= nperel;i < n; ++i) {
        float lhv = lhs[i] + lhinc;
        float rhv = rhs[i] + rhinc;
        float miv = lhv * lambda + rhv * m1l;
        ret += lambdav * lhv * logf(lh / miv)
             + m1l * rhv * logf(rhv / miv);
    }
#else
    double lhsum = 0., rhsum = 0.;
    for(;i < n; ++i) {
        float lhv = lhs[i] + lhinc;
        float rhv = rhs[i] + rhinc;
        float miv = lambda * lhv + rhv * m1l;
        lhsum += lhv * logf(lhv / miv);
        rhsum += rhv * logf(rhv / miv);
    }
    ret = lambda * lhsum + m1l * rhsum;
#endif
    return ret;
}

double __llr_reduce_aligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, size_t n, double lambda, double lhinc, double rhinc)
{
    const double m1l = 1. - lambda;
    double ret = 0.;
    size_t i = 0;
#if __AVX512F__
    const size_t nperel = sizeof(__m512d) / sizeof(double);
    __m512 lhsum = _mm512_setzero_pd(), rhsum = _mm512_setzero_pd();
    size_t nsimd = n / nperel;
    //
    __m512d vlambda = _mm512_set1_pd(lambda), vm1l = _mm512_set1_pd(m1l);
    __m512d lhv = _mm512_set1_pd(lhinc);
    __m512d rhv = _mm512_set1_pd(rhinc);
    #pragma GCC unroll 4
    for(i = 0; i < nsimd; ++i) {
        __m512d lhl = _mm512_add_pd(_mm512_load_pd(&lhs[i * nperel]), lhv);
        __m512d rhl = _mm512_add_pd(_mm512_load_pd(&rhs[i * nperel]), rhv);
#if DIV1
        __m512d mv0 = _mm512_div_pd(_mm512_set1_pd(1.), _mm512_add_pd(_mm512_mul_pd(vlambda, lhl), _mm512_mul_pd(vm1l, rhl)));
        __m512d lv0 = _mm512_mul_pd(lhl, Sleef_logd8_u35(_mm512_mul_pd(lhl, mv)));
        __m512d rv0 = _mm512_mul_pd(rhl, Sleef_logd8_u35(_mm512_mul_pd(rhl, mv)));
#elif LOG3
        __m512d lmv0 = Sleef_logd8_u35(_mm512_add_pd(_mm512_mul_pd(vlambda, lhl), _mm512_mul_pd(vm1l, rhl)));
        __m512d lv0 = _mm512_mul_pd(lhl, _mm512_sub_pd(Sleef_logd8_u35(lhl), lmv0));
        __m512d rv0 = _mm512_mul_pd(rhl, _mm512_sub_pd(Sleef_logd8_u35(rhl), rmv0));
#else
        __m512d mv0 = _mm512_add_pd(_mm512_mul_pd(vlambda, lhl), _mm512_mul_pd(vm1l, rhl));
        __m512d lv0 = _mm512_mul_pd(lhl, Sleef_logd8_u35(_mm512_div_pd(lhl, mv)));
        __m512d rv0 = _mm512_mul_pd(rhl, Sleef_logd8_u35(_mm512_div_pd(rhl, mv)));
#endif
        lhsum = _mm512_add_pd(lhsum, lv0);
        rhsum = _mm512_add_pd(rhsum, rv0);
    }
    ret += lambda * _mm512_reduce_add_pd(lhsum) + m1l * _mm512_reduce_add_pd(rhsum);
    for(i *= nperel;i < n; ++i) {
        double xv = lhs[i] + lhinc;
        double yv = rhs[i] + rhinc;
        double mnv = (xv + yv) * .5;
        ret += lambda * xv * log(xv / mnv) + m1l * rhs[i] * log(yv / mnv);
    }
#elif __AVX2__
    const size_t nperel = sizeof(__m256d) / sizeof(double);
    __m256d lhsum = _mm256_setzero_pd(), rhsum = _mm256_setzero_pd();
    size_t nsimd = n / nperel;
    //
    __m256d lhv = _mm256_set1_pd(lhinc), rhv = _mm256_set1_pd(rhinc);
    __m256d vlambda = _mm256_set1_pd(lambda), vm1l = _mm256_set1_pd(m1l);
    for(; i < nsimd; ++i) {
        __m256d lhsa = _mm256_add_pd(_mm256_load_pd(&lhs[i * nperel]), lhv);
        __m256d rhsa = _mm256_add_pd(_mm256_load_pd(&rhs[i * nperel]), rhv);
#if DIV1
        __m256d mv = _mm256_div_pd(_mm256_set1_pd(1.), _mm256_add_pd(_mm256_mul_pd(vlambda, lhsa), _mm256_mul_pd(vm1l, rhsa)));
        lhsum = _mm256_add_pd(lhsum, _mm256_mul_pd(lhsa, Sleef_logd4_u35(_mm256_mul_pd(lhsa, mv))));
        rhsum = _mm256_add_pd(rhsum, _mm256_mul_pd(rhsa, Sleef_logd4_u35(_mm256_mul_pd(rhsa, mv))));
#elif LOG3
        __m256d lmv = Sleef_logd4_u35(_mm256_add_pd(_mm256_mul_pd(vlambda, lhsa), _mm256_mul_pd(vm1l, rhsa)));
        lhsum = _mm256_add_pd(lhsum, _mm256_mul_pd(lhsa, _mm256_sub_pd(Sleef_logd4_u35(lhsa), lmv)));
        rhsum = _mm256_add_pd(rhsum, _mm256_mul_pd(rhsa, _mm256_sub_pd(Sleef_logd4_u35(rhsa), lmv)));
#else
        __m256d mv = _mm256_add_pd(_mm256_mul_pd(vlambda, lhsa), _mm256_mul_pd(vm1l, rhsa));
        lhsum = _mm256_add_pd(lhsum, _mm256_mul_pd(lhsa, Sleef_logd4_u35(_mm256_div_pd(lhsa, mv))));
        rhsum = _mm256_add_pd(rhsum, _mm256_mul_pd(rhsa, Sleef_logd4_u35(_mm256_mul_pd(rhsa, mv))));
#endif
    }
    ret += lambda * hsum_double_avx(lhsum) + m1l * hsum_double_avx(rhsum);
    for(i *= nperel;i < n; ++i) {
        double lhv = lhs[i] + lhinc;
        double rhv = rhs[i] + rhinc;
        double miv = lhv * lambda + rhv * m1l;
        ret += lambda * lhv * log(lhv / miv)
             + m1l * rhv * log(rhv / miv);
    }
#elif __SSE2__
    const size_t nperel = sizeof(__m128d) / sizeof(double);
    __m128d lhsum = _mm_setzero_pd(), rhsum = _mm_setzero_pd();
    size_t nsimd = n / nperel;
    //
    __m128d vlambda = _mm_set1_pd(lambda), vm1l = _mm_set1_pd(m1l);
    for(; i < nsimd; ++i) {
        __m128d lhsa = _mm_add_pd(_mm_load_pd(&lhs[i * nperel]), lhinc);
        __m128d rhsa = _mm_add_pd(_mm_load_pd(&rhs[i * nperel]), rhinc);
#if DIV1
        __m128d mv = _mm_div_pd(_mm_set1_pd(1.), _mm_add_pd(_mm_mul_pd(vlambda, lhsa), _mm_mul_pd(vm1l, rhsa)));
        lhsum = _mm_add_pd(lhsum, _mm_mul_pd(lhsa, Sleef_logd2_u35(_mm_mul_pd(lhsa, mv))));
        rhsum = _mm_add_pd(rhsum, _mm_mul_pd(rhsa, Sleef_logd2_u35(_mm_mul_pd(rhsa, mv))));
#elif LOG3
        __m128d lmv = Sleef_logd2_u35(_mm_add_pd(_mm_mul_pd(vlambda, lhsa), _mm_mul_pd(vm1l, rhsa)));
        lhsum = _mm_add_pd(lhsum, _mm_mul_pd(lhsa, _mm_sub_pd(Sleef_logd2_u35(lhsa, lmv))));
        rhsum = _mm_add_pd(rhsum, _mm_mul_pd(rhsa, _mm_sub_pd(Sleef_logd2_u35(rhsa, lmv))));
#else
        __m128d mv = _mm_add_pd(_mm_mul_pd(vlambda, lhsa), _mm_mul_pd(vm1l, rhsa));
        lhsum = _mm_add_pd(lhsum, _mm_mul_pd(lhsa, Sleef_logd2_u35(_mm_div_pd(lhsa, mv))));
        rhsum = _mm_add_pd(rhsum, _mm_mul_pd(rhsa, Sleef_logd2_u35(_mm_div_pd(rhsa, mv))));
#endif
    }
    ret += lambda * (lhsum[0] + lhsum[1]) + m1l * (rhsum[0] + rhsum[1]);
    for(i *= nperel;i < n; ++i) {
        double lhv = lhs[i] + lhinc;
        double rhv = rhs[i] + rhinc;
        double miv = lhv * lambda + rhv * m1l;
        ret += lambdav * lhv * log(lh / miv)
             + m1l * rhv * log(rhv / miv);
    }
#else
    double lhsum = 0., rhsum = 0.;
    for(;i < n; ++i) {
        double lhv = lhs[i] + lhinc;
        double rhv = rhs[i] + rhinc;
        double miv = lambda * lhv + rhv * m1l;
        lhsum += lhv * log(lhv / miv);
        rhsum += rhv * log(rhv / miv);
    }
    ret = lambda * lhsum + m1l * rhsum;
#endif
    return ret;
}

double __kl_reduce_unaligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhi, double rhi) {
    double ret = 0.;
    size_t i = 0;
#if __AVX512F__
#define lhload(i) _mm512_add_pd(_mm512_loadu_pd(lhs + ((i) * nper)), lhv)
#define rhload(i) _mm512_add_pd(_mm512_loadu_pd(rhs + ((i) * nper)), rhv)
    __m512d lhv = _mm512_set1_pd(lhi), rhv = _mm512_set1_pd(rhi);
    assert((uint64_t *)(lhs) % 64 == 0);
    const size_t nper = sizeof(__m512) / sizeof(double);
    const size_t nsimd = n / nper;
    const size_t nsimd4 = (nsimd / 4) * 4;
    {
        __m512d sum = _mm512_setzero_pd();
        for(; i < nsimd4; i += 4) {
            __m512d lh0 = lhload(i), lh1 = lhload(i + 1), lh2 = lhload(i + 2); lh3 = lhload(i + 3);
            __m512d rh0 = rhload(i), rh1 = rhload(i + 1), rh2 = rhload(i + 2); rh3 = rhload(i + 3);
            __m512d v0 = _mm512_mul_pd(lh0, Sleef_logd8_u35(_mm512_div_pd(lh0, rh0)));
            __m512d v1 = _mm512_mul_pd(lh1, Sleef_logd8_u35(_mm512_div_pd(lh1, rh1)));
            __m512d v2 = _mm512_mul_pd(lh2, Sleef_logd8_u35(_mm512_div_pd(lh2, rh2)));
            __m512d v3 = _mm512_mul_pd(lh3, Sleef_logd8_u35(_mm512_div_pd(lh3, rh3)));
            sum = _mm512_add_pd(sum,  _mm512_add_pd(_mm512_add_pd(v0, v1), _mm512_add_pd(v2, v3)));
        }
        ret += _mm512_reduce_add_pd(sum);
    }
    for(; i < nsimd; ++i) {
        __m512d lh = lhload(i), rh = rhload(i);
#undef lhload
#undef rhload
        __m512d logv = Sleef_logd8_u35(_mm512_div_pd(lh, rhload(i)));
        ret += _mm512_reduce_add_pd(_mm512_mul_pd(lh, logv));
    }
    i *= nper;
#elif __AVX2__
    assert(((uint64_t)lhs) % 32 == 0);
    const size_t nper = sizeof(__m256) / sizeof(double);
    const size_t nsimd = n / nper;
    const size_t nsimd4 = (nsimd / 4) * 4;
#define lhload(i) _mm256_add_pd(_mm256_loadu_pd(lhs + ((i) * nper)), lhv)
#define rhload(i) _mm256_add_pd(_mm256_loadu_pd(rhs + ((i) * nper)), rhv)
    __m256d lhv = _mm256_set1_pd(lhi), rhv = _mm256_set1_pd(rhi);
    __m256d sum = _mm256_setzero_pd();
    for(; i < nsimd4; i += 4) {
        __m256d lh0 = lhload(i), lh1 = lhload(i + 1), lh2 = lhload(i + 2), lh3 = lhload(i + 3);
        __m256d rh0 = rhload(i), rh1 = rhload(i + 1), rh2 = rhload(i + 2), rh3 = rhload(i + 3);
        __m256d v0 = _mm256_mul_pd(lh0, Sleef_logd4_u35(_mm256_div_pd(lh0, rh0)));
        __m256d v1 = _mm256_mul_pd(lh1, Sleef_logd4_u35(_mm256_div_pd(lh1, rh1)));
        __m256d v2 = _mm256_mul_pd(lh2, Sleef_logd4_u35(_mm256_div_pd(lh2, rh2)));
        __m256d v3 = _mm256_mul_pd(lh3, Sleef_logd4_u35(_mm256_div_pd(lh3, rh3)));
        sum = _mm256_add_pd(sum,  _mm256_add_pd(_mm256_add_pd(v0, v1), _mm256_add_pd(v2, v3)));
    }
    for(; i < nsimd; ++i) {
        __m256d lh = lhload(i), rh = rhload(i);
#undef rhload
#undef lhload
        __m256d res = _mm256_mul_pd(lh, Sleef_logd4_u35(_mm256_div_pd(lh, rh)));
        sum = _mm256_add_pd(sum, res);
    }
    ret = hsum_double_avx(sum);
    i *= nper;
#elif __SSE2__
    assert(((uint64_t)lhs) % 16 == 0);
    const size_t nper = sizeof(__m128) / sizeof(double);
    const size_t nsimd = n / nper;
    __m128d lhv = _mm_set1_pd(lhi), rhv = _mm_set1_pd(rhi);
#define lhload(i) _mm_add_pd(_mm_loadu_pd(lhs + ((i) * nper)), lhv)
#define rhload(i) _mm_add_pd(_mm_loadu_pd(rhs + ((i) * nper)), rhv)
    {
        __m128d v = _mm_setzero_pd();
        #pragma GCC unroll 4
        for(; i < nsimd; ++i) {
            __m128d lh = lhload(i), rh = rhload(i);
#undef lhload
#undef rhload
            v = _mm_add_pd(v, _mm_mul_pd(lh, Sleef_logd2_u35(_mm_div_pd(lh, rh))));
        }
        i *= nper;
        ret = v[0] + v[1];
    }
#endif
    #pragma GCC unroll 8
    for(; i < n; ++i) {
        float lhv = lhs[i] + lhi, rhv = rhs[i] + rhi;
        ret += lhv * logf(lhv / rhv);
    }
#ifndef NDEBUG
    double oret = 0.;
    for(size_t j = 0; j < n; ++j) {
        float lhv = lhs[i] + lhi, rhv = rhs[i] + rhi;
        oret += lhv * logf(lhv / rhv);
    }
    assert(fabs(oret - ret) < 1e-5);
#endif
    return ret;
}

double __kl_reduce_unaligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhi, float rhi) {
    double ret = 0.;
    size_t i = 0;
#if __AVX512F__
    assert(((uint64_t)lhs) % 64 == 0);
    const size_t nper = sizeof(__m512) / sizeof(float);
    const size_t nsimd = n / nper;
    const size_t nsimd4 = (nsimd / 4) * 4;
    __m512 sum = _mm512_setzero_ps();
    #pragma GCC unroll 4
    for(; i < nsimd; ++i) {
        __m512 lh = _mm512_add_ps(_mm512_loadu_ps(lhs + (i * nper)), _mm512_set1_ps(lhi));
        __m512 rh = _mm512_add_ps(_mm512_loadu_ps(rhs + (i * nper)), _mm512_set1_ps(rhi));
        __m512 div = _mm512_div_ps(lh, rh);
        __m512 logv = Sleef_logf16_u35(div);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(lh, logv));
    }
    ret = _mm512_reduce_add_ps(sum);
    i *= nper;
#elif __AVX2__
    assert(((uint64_t)lhs) % 32 == 0);
    const size_t nper = sizeof(__m256) / sizeof(float);
    const size_t nsimd = n / nper;

    __m256 sum = _mm256_setzero_ps();
    #pragma GCC unroll 4
    for(; i < nsimd; ++i) {
        __m256 lh = _mm256_add_ps(_mm256_loadu_ps(lhs + (i * nper)), _mm256_set1_ps(lhi));
        __m256 rh = _mm256_add_ps(_mm256_loadu_ps(rhs + (i * nper)), _mm256_set1_ps(rhi));
        __m256 res = _mm256_mul_ps(lh, Sleef_logf8_u35(_mm256_div_ps(lh, rh)));
        sum = _mm256_add_ps(sum, res);
    }
    ret += broadcast_reduce_add_si256_ps(sum)[0];
    i *= nper;
#elif __SSE2__
    assert(reinterpret_cast<uint64_t>(lhs) % 16 == 0);
    const size_t nper = sizeof(__m128) / sizeof(float);
    const size_t nsimd = n / nper;
    __m128 v = _mm_setzero_ps();
    #pragma GCC unroll 4
    for(; i < nsimd; ++i) {
        __m128 lh = _mm_add_ps(_mm_loadu_ps(lhs + (i * nper)), _mm_set1_ps(lhi));
        __m128 rh = _mm_add_ps(_mm_loadu_ps(rhs + (i * nper)), _mm_set1_ps(rhi));
        __m128 div = _mm_div_ps(lh, rh);
        __m128 logv = Sleef_logf4_u35(div);
        __m128 res = _mm_mul_ps(lh, logv);
        v = _mm_add_ps(v, res);
    }
    ret += v[0]; ret += v[1]; ret += v[2]; ret += v[3];
    i *= nper;
#endif
    #pragma GCC unroll 8
    for(; i < n; ++i) {
        float lhv = lhs[i] + lhi;
        float rhv = rhs[i] + rhi;
        ret += lhv * logf(lhv / rhv);
    }
#ifndef NDEBUG
    double oret = 0.;
    for(size_t j = 0; j < n; ++j) {
        float lhv = lhs[i] + lhi;
        float rhv = rhs[i] + rhi;
        oret += lhv * logf(lhv / rhv);
    }
    assert(fabs(oret - ret) < 1e-5);
#endif
    return ret;
}

double __llr_reduce_unaligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, size_t n, double lambda, float lhinc, float rhinc) {
    const double m1l = 1. - lambda;
    double ret = 0.;
    size_t i = 0;
#if __AVX512F__
    const size_t nperel = sizeof(__m512) / sizeof(float);
    __m512 lhsum = _mm512_setzero_ps(), rhsum = _mm512_setzero_ps();
    size_t nsimd = n / nperel;
    //
    __m512 vlambda = _mm512_set1_ps(lambda), vm1l = _mm512_set1_ps(m1l);
    __m512 lhv = _mm512_set1_ps(lhinc);
    __m512 rhv = _mm512_set1_ps(rhinc);
    #pragma GCC unroll 4
    for(i = 0; i < nsimd; ++i) {
        __m512 lhl = _mm512_add_ps(_mm512_loadu_ps(&lhs[i * nperel]), lhv);
        __m512 rhl = _mm512_add_ps(_mm512_loadu_ps(&rhs[i * nperel]), rhv);
#if DIV1
        __m512 mv0 = _mm512_div_ps(_mm512_set1_ps(1.), _mm512_add_ps(_mm512_mul_ps(vlambda, lhl), _mm512_mul_ps(vm1l, rhl)));
        __m512 lv0 = _mm512_mul_ps(lhl, Sleef_logf16_u35(_mm512_mul_ps(lhl, mv)));
        __m512 rv0 = _mm512_mul_ps(rhl, Sleef_logf16_u35(_mm512_mul_ps(rhl, mv)));
#elif LOG3
        __m512 lmv0 = Sleef_logf16_u35(_mm512_add_ps(_mm512_mul_ps(vlambda, lhl), _mm512_mul_ps(vm1l, rhl)));
        __m512 lv0 = _mm512_mul_ps(lhl, _mm512_sub_ps(Sleef_logf16_u35(lhl), lmv0));
        __m512 rv0 = _mm512_mul_ps(rhl, _mm512_sub_ps(Sleef_logf16_u35(rhl), rmv0));
#else
        __m512 mv0 = _mm512_add_ps(_mm512_mul_ps(vlambda, lhl), _mm512_mul_ps(vm1l, rhl));
        __m512 lv0 = _mm512_mul_ps(lhl, Sleef_logf16_u35(_mm512_div_ps(lhl, mv)));
        __m512 rv0 = _mm512_mul_ps(rhl, Sleef_logf16_u35(_mm512_div_ps(rhl, mv)));
#endif
        lhsum = _mm512_add_ps(lhsum, lv0);
        rhsum = _mm512_add_ps(rhsum, rv0);
    }
    ret += lambda * _mm512_reduce_add_ps(lhsum) + m1l * _mm512_reduce_add_ps(rhsum);
    for(i *= nperel;i < n; ++i) {
        float xv = lhs[i] + lhinc;
        float yv = rhs[i] + rhinc;
        float mnv = (xv + yv) * .5;
        ret += lambda * xv * logf(xv / mnv) + m1l * rhs[i] * logf(yv / mnv);
    }
#elif __AVX2__
    const size_t nperel = sizeof(__m256) / sizeof(float);
    __m256 lhsum = _mm256_setzero_ps(), rhsum = _mm256_setzero_ps();
    size_t nsimd = n / nperel;
    __m256 lhv = _mm256_set1_ps(lhinc);
    __m256 rhv = _mm256_set1_ps(rhinc);
    __m256 vlambda = _mm256_set1_ps(lambda), vm1l = _mm256_set1_ps(m1l);
    for(; i < nsimd; ++i) {
        __m256 lhsa = _mm256_add_ps(_mm256_loadu_ps(&lhs[i * nperel]), lhv);
        __m256 rhsa = _mm256_add_ps(_mm256_loadu_ps(&rhs[i * nperel]), rhv);
#if DIV1
        __m256 mv = _mm256_div_ps(_mm256_set1_ps(1.), _mm256_add_ps(_mm256_mul_ps(vlambda, lhsa), _mm256_mul_ps(vm1l, rhsa)));
        lhsum = _mm256_add_ps(lhsum, _mm256_mul_ps(lhsa, Sleef_logf8_u35(_mm256_mul_ps(lhsa, mv))));
        rhsum = _mm256_add_ps(rhsum, _mm256_mul_ps(rhsa, Sleef_logf8_u35(_mm256_mul_ps(rhsa, mv))));
#elif LOG3
        __m256 lmv = Sleef_logf8_u35(_mm256_add_ps(_mm256_mul_ps(vlambda, lhsa), _mm256_mul_ps(vm1l, rhsa)));
        lhsum = _mm256_add_ps(lhsum, _mm256_mul_ps(lhsa, _mm256_sub_ps(Sleef_logf8_u35(lhsa), lmv)));
        rhsum = _mm256_add_ps(rhsum, _mm256_mul_ps(rhsa, _mm256_sub_ps(Sleef_logf8_u35(rhsa), lmv)));
#else
        __m256 mv = _mm256_add_ps(_mm256_mul_ps(vlambda, lhsa), _mm256_mul_ps(vm1l, rhsa));
        lhsum = _mm256_add_ps(lhsum, _mm256_mul_ps(lhsa, Sleef_logf8_u35(_mm256_div_ps(lhsa, mv))));
        rhsum = _mm256_add_ps(rhsum, _mm256_mul_ps(rhsa, Sleef_logf8_u35(_mm256_mul_ps(rhsa, mv))));
#endif
    }
    ret += lambda * broadcast_reduce_add_si256_ps(lhsum)[0] + m1l * broadcast_reduce_add_si256_ps(rhsum)[0];
    for(i *= nperel;i < n; ++i) {
        float lhv = lhs[i] + lhinc;
        float rhv = rhs[i] + rhinc;
        float miv = lhv * lambda + rhv * m1l;
        ret += lambda * lhv * logf(lhv / miv)
             + m1l * rhv * logf(rhv / miv);
    }
#elif __SSE2__
    const size_t nperel = sizeof(__m128) / sizeof(float);
    __m128 lhsum = _mm_setzero_ps(), rhsum = _mm_setzero_ps();
    size_t nsimd = n / nperel;
    size_t nsimd4 = nsimd / 4;
    //
    __m128 vlambda = _mm_set1_ps(lambda), vm1l = _mm_set1_ps(m1l);
    for(; i < nsimd; ++i) {
        __m128 lhsa = _mm_add_ps(_mm_loadu_ps(&lhs[i * nperel]), lhinc);
        __m128 rhsa = _mm_add_ps(_mm_loadu_ps(&rhs[i * nperel]), rhinc);
#if DIV1
        __m128 mv = _mm_div_ps(_mm_set1_ps(1.), _mm_add_ps(_mm_mul_ps(vlambda, lhsa0), _mm_mul_ps(vm1l, rhsa0)));
        lhsum = _mm_add_ps(lhsum, _mm_mul_ps(lhsa, Sleef_logf4_u35(_mm_mul_ps(lhsa, mv))));
        rhsum = _mm_add_ps(rhsum, _mm_mul_ps(rhsa, Sleef_logf4_u35(_mm_mul_ps(rhsa, mv))));
#elif LOG3
        __m128 lmv = Sleef_logf4_u35(_mm_add_ps(_mm_mul_ps(vlambda, lhsa0), _mm_mul_ps(vm1l, rhsa0)));
        lhsum = _mm_add_ps(lhsum, _mm_mul_ps(lhsa, _mm_sub_ps(Sleef_logf4_u35(lhsa, lmv))));
        rhsum = _mm_add_ps(rhsum, _mm_mul_ps(rhsa, _mm_sub_ps(Sleef_logf4_u35(rhsa, lmv))));
#else
        __m128 mv = _mm_add_ps(_mm_mul_ps(vlambda, lhsa0), _mm_mul_ps(vm1l, rhsa0));
        lhsum = _mm_add_ps(lhsum, _mm_mul_ps(lhsa, Sleef_logf4_u35(_mm_div_ps(lhsa, mv))));
        rhsum = _mm_add_ps(rhsum, _mm_mul_ps(rhsa, Sleef_logf4_u35(_mm_div_ps(rhsa, mv))));
#endif
    }
    ret += lambda * broadcast_reduce_add_si128_ps(lhsum) + m1l * broadcast_reduce_add_si128_ps(rhsum);
    for(i *= nperel;i < n; ++i) {
        float lhv = lhs[i] + lhinc;
        float rhv = rhs[i] + rhinc;
        float miv = lhv * lambda + rhv * m1l;
        ret += lambdav * lhv * logf(lh / miv)
             + m1l * rhv * logf(rhv / miv);
    }
#else
    double lhsum = 0., rhsum = 0.;
    for(;i < n; ++i) {
        float lhv = lhs[i] + lhinc;
        float rhv = rhs[i] + rhinc;
        float miv = lambda * lhv + rhv * m1l;
        lhsum += lhv * logf(lhv / miv);
        rhsum += rhv * logf(rhv / miv);
    }
    ret = lambda * lhsum + m1l * rhsum;
#endif
    return ret;
}

double __llr_reduce_unaligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, size_t n, double lambda, double lhinc, double rhinc)
{
    const double m1l = 1. - lambda;
    double ret = 0.;
    size_t i = 0;
#if __AVX512F__
    const size_t nperel = sizeof(__m512d) / sizeof(double);
    __m512 lhsum = _mm512_setzero_pd(), rhsum = _mm512_setzero_pd();
    size_t nsimd = n / nperel;
    //
    __m512d vlambda = _mm512_set1_pd(lambda), vm1l = _mm512_set1_pd(m1l);
    __m512d lhv = _mm512_set1_pd(lhinc);
    __m512d rhv = _mm512_set1_pd(rhinc);
    #pragma GCC unroll 4
    for(i = 0; i < nsimd; ++i) {
        __m512d lhl = _mm512_add_pd(_mm512_loadu_pd(&lhs[i * nperel]), lhv);
        __m512d rhl = _mm512_add_pd(_mm512_loadu_pd(&rhs[i * nperel]), rhv);
#if DIV1
        __m512d mv0 = _mm512_div_pd(_mm512_set1_pd(1.), _mm512_add_pd(_mm512_mul_pd(vlambda, lhl), _mm512_mul_pd(vm1l, rhl)));
        __m512d lv0 = _mm512_mul_pd(lhl, Sleef_logd8_u35(_mm512_mul_pd(lhl, mv)));
        __m512d rv0 = _mm512_mul_pd(rhl, Sleef_logd8_u35(_mm512_mul_pd(rhl, mv)));
#elif LOG3
        __m512d lmv0 = Sleef_logd8_u35(_mm512_add_pd(_mm512_mul_pd(vlambda, lhl), _mm512_mul_pd(vm1l, rhl)));
        __m512d lv0 = _mm512_mul_pd(lhl, _mm512_sub_pd(Sleef_logd8_u35(lhl), lmv0));
        __m512d rv0 = _mm512_mul_pd(rhl, _mm512_sub_pd(Sleef_logd8_u35(rhl), rmv0));
#else
        __m512d mv0 = _mm512_add_pd(_mm512_mul_pd(vlambda, lhl), _mm512_mul_pd(vm1l, rhl));
        __m512d lv0 = _mm512_mul_pd(lhl, Sleef_logd8_u35(_mm512_div_pd(lhl, mv)));
        __m512d rv0 = _mm512_mul_pd(rhl, Sleef_logd8_u35(_mm512_div_pd(rhl, mv)));
#endif
        lhsum = _mm512_add_pd(lhsum, lv0);
        rhsum = _mm512_add_pd(rhsum, rv0);
    }
    ret += lambda * _mm512_reduce_add_pd(lhsum) + m1l * _mm512_reduce_add_pd(rhsum);
    for(i *= nperel;i < n; ++i) {
        double xv = lhs[i] + lhinc;
        double yv = rhs[i] + rhinc;
        double mnv = (xv + yv) * .5;
        ret += lambda * xv * log(xv / mnv) + m1l * rhs[i] * log(yv / mnv);
    }
#elif __AVX2__
    const size_t nperel = sizeof(__m256d) / sizeof(double);
    __m256d lhsum = _mm256_setzero_pd(), rhsum = _mm256_setzero_pd();
    size_t nsimd = n / nperel;
    //
    __m256d lhv = _mm256_set1_pd(lhinc), rhv = _mm256_set1_pd(rhinc);
    __m256d vlambda = _mm256_set1_pd(lambda), vm1l = _mm256_set1_pd(m1l);
    for(; i < nsimd; ++i) {
        __m256d lhsa = _mm256_add_pd(_mm256_loadu_pd(&lhs[i * nperel]), lhv);
        __m256d rhsa = _mm256_add_pd(_mm256_loadu_pd(&rhs[i * nperel]), rhv);
#if DIV1
        __m256d mv = _mm256_div_pd(_mm256_set1_pd(1.), _mm256_add_pd(_mm256_mul_pd(vlambda, lhsa), _mm256_mul_pd(vm1l, rhsa)));
        lhsum = _mm256_add_pd(lhsum, _mm256_mul_pd(lhsa, Sleef_logd4_u35(_mm256_mul_pd(lhsa, mv))));
        rhsum = _mm256_add_pd(rhsum, _mm256_mul_pd(rhsa, Sleef_logd4_u35(_mm256_mul_pd(rhsa, mv))));
#elif LOG3
        __m256d lmv = Sleef_logd4_u35(_mm256_add_pd(_mm256_mul_pd(vlambda, lhsa), _mm256_mul_pd(vm1l, rhsa)));
        lhsum = _mm256_add_pd(lhsum, _mm256_mul_pd(lhsa, _mm256_sub_pd(Sleef_logd4_u35(lhsa), lmv)));
        rhsum = _mm256_add_pd(rhsum, _mm256_mul_pd(rhsa, _mm256_sub_pd(Sleef_logd4_u35(rhsa), lmv)));
#else
        __m256d mv = _mm256_add_pd(_mm256_mul_pd(vlambda, lhsa), _mm256_mul_pd(vm1l, rhsa));
        lhsum = _mm256_add_pd(lhsum, _mm256_mul_pd(lhsa, Sleef_logd4_u35(_mm256_div_pd(lhsa, mv))));
        rhsum = _mm256_add_pd(rhsum, _mm256_mul_pd(rhsa, Sleef_logd4_u35(_mm256_mul_pd(rhsa, mv))));
#endif
    }
    ret += lambda * hsum_double_avx(lhsum) + m1l * hsum_double_avx(rhsum);
    for(i *= nperel;i < n; ++i) {
        double lhv = lhs[i] + lhinc;
        double rhv = rhs[i] + rhinc;
        double miv = lhv * lambda + rhv * m1l;
        ret += lambda * lhv * log(lhv / miv)
             + m1l * rhv * log(rhv / miv);
    }
#elif __SSE2__
    const size_t nperel = sizeof(__m128d) / sizeof(double);
    __m128d lhsum = _mm_setzero_pd(), rhsum = _mm_setzero_pd();
    size_t nsimd = n / nperel;
    //
    __m128d vlambda = _mm_set1_pd(lambda), vm1l = _mm_set1_pd(m1l);
    for(; i < nsimd; ++i) {
        __m128d lhsa = _mm_add_pd(_mm_loadu_pd(&lhs[i * nperel]), lhinc);
        __m128d rhsa = _mm_add_pd(_mm_loadu_pd(&rhs[i * nperel]), rhinc);
#if DIV1
        __m128d mv = _mm_div_pd(_mm_set1_pd(1.), _mm_add_pd(_mm_mul_pd(vlambda, lhsa), _mm_mul_pd(vm1l, rhsa)));
        lhsum = _mm_add_pd(lhsum, _mm_mul_pd(lhsa, Sleef_logd2_u35(_mm_mul_pd(lhsa, mv))));
        rhsum = _mm_add_pd(rhsum, _mm_mul_pd(rhsa, Sleef_logd2_u35(_mm_mul_pd(rhsa, mv))));
#elif LOG3
        __m128d lmv = Sleef_logd2_u35(_mm_add_pd(_mm_mul_pd(vlambda, lhsa), _mm_mul_pd(vm1l, rhsa)));
        lhsum = _mm_add_pd(lhsum, _mm_mul_pd(lhsa, _mm_sub_pd(Sleef_logd2_u35(lhsa, lmv))));
        rhsum = _mm_add_pd(rhsum, _mm_mul_pd(rhsa, _mm_sub_pd(Sleef_logd2_u35(rhsa, lmv))));
#else
        __m128d mv = _mm_add_pd(_mm_mul_pd(vlambda, lhsa), _mm_mul_pd(vm1l, rhsa));
        lhsum = _mm_add_pd(lhsum, _mm_mul_pd(lhsa, Sleef_logd2_u35(_mm_div_pd(lhsa, mv))));
        rhsum = _mm_add_pd(rhsum, _mm_mul_pd(rhsa, Sleef_logd2_u35(_mm_div_pd(rhsa, mv))));
#endif
    }
    ret += lambda * (lhsum[0] + lhsum[1]) + m1l * (rhsum[0] + rhsum[1]);
    for(i *= nperel;i < n; ++i) {
        double lhv = lhs[i] + lhinc;
        double rhv = rhs[i] + rhinc;
        double miv = lhv * lambda + rhv * m1l;
        ret += lambdav * lhv * log(lh / miv)
             + m1l * rhv * log(rhv / miv);
    }
#else
    double lhsum = 0., rhsum = 0.;
    for(;i < n; ++i) {
        double lhv = lhs[i] + lhinc;
        double rhv = rhs[i] + rhinc;
        double miv = lambda * lhv + rhv * m1l;
        lhsum += lhv * log(lhv / miv);
        rhsum += rhv * log(rhv / miv);
    }
    ret = lambda * lhsum + m1l * rhsum;
#endif
    return ret;
}
