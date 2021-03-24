#include "x86intrin.h"
#include "sleef.h"
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include "libkl.h"
#include <stdio.h>



// LLR has 2 versions:
// default, and LOG3
// LOG3 may have better numerical performance, while default will be faster


#ifdef USE_APPROX_LOG
    #define Sleef_logd2_u35 _mm_lkl_alog_pd
    #define Sleef_logd4_u35 _mm256_lkl_alog_pd
    #define Sleef_logd8_u35 _mm512_lkl_alog_pd
    #define Sleef_logf4_u35 _mm_lkl_alog_ps
    #define Sleef_logf8_u35 _mm256_lkl_alog_ps
    #define Sleef_logf16_u35 _mm512_lkl_alog_ps
#endif


#if !__AVX512DQ__
#ifndef _mm512_cvtepi64_pd
  #define _mm512_cvtepi64_pd(x) _mm512_fmadd_pd(\
          _mm512_cvtepu32_pd(_mm512_cvtepi64_epi32(_mm512_srli_epi64(x, 32))),\
          _mm512_set1_pd(0x100000000LL), _mm512_cvtepu32_pd(_mm512_cvtepi64_epi32(x)))
#endif
#endif

#ifdef LIBKL_HIGH_PRECISION
#  ifndef Sleef_logd2_u35
#    define Sleef_logd2_u35 Sleef_logd2_u10
#  endif
#  ifndef Sleef_logd4_u35
#    define Sleef_logd4_u35 Sleef_logd4_u10
#  endif
#  ifndef Sleef_logd8_u35
#    define Sleef_logd8_u35 Sleef_logd8_u10
#  endif
#  ifndef Sleef_logf4_u35
#    define Sleef_logf4_u35 Sleef_logf4_u10
#  endif
#  ifndef Sleef_logf8_u35
#    define Sleef_logf8_u35 Sleef_logf8_u10
#  endif
#  ifndef Sleef_logf16_u35
#    define Sleef_logf16_u35 Sleef_logf16_u10
#  endif
#  ifndef Sleef_sqrtd2_u35
#    define Sleef_sqrtd2_u35 Sleef_sqrtd2_u05
#  endif
#  ifndef Sleef_sqrtd4_u35
#    define Sleef_sqrtd4_u35 Sleef_sqrtd4_u05
#  endif
#  ifndef Sleef_sqrtd8_u35
#    define Sleef_sqrtd8_u35 Sleef_sqrtd8_u05
#  endif
#  ifndef Sleef_sqrtf4_u35
#    define Sleef_sqrtf4_u35 Sleef_sqrtf4_u05
#  endif
#  ifndef Sleef_sqrtf8_u35
#    define Sleef_sqrtf8_u35 Sleef_sqrtf8_u05
#  endif
#  ifndef Sleef_sqrtf16_u35
#    define Sleef_sqrtf16_u35 Sleef_sqrtf16_u05
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if __AVX512F__
static inline __attribute__((always_inline)) float _mm512_reduce_add_psf(__m512 x) {
    return _mm512_reduce_add_ps(x);
}
static inline __attribute__((always_inline)) double _mm512_reduce_add_pdd(__m512d x) {
    return _mm512_reduce_add_pd(x);
}
#endif


#ifdef __AVX2__
static inline __attribute__((always_inline)) __m256 broadcast_reduce_add_si256_ps(__m256 x) {
    const __m256 permHalves = _mm256_permute2f128_ps(x, x, 1);
    const __m256 m0 = _mm256_add_ps(permHalves, x);
    const __m256 perm0 = _mm256_permute_ps(m0, 0b01001110);
    const __m256 m1 = _mm256_add_ps(m0, perm0);
    const __m256 perm1 = _mm256_permute_ps(m1, 0b10110001);
    const __m256 m2 = _mm256_add_ps(perm1, m1);
    return m2;
}
static inline float broadcast_reduce_add_si256_psf(__m256 x) {
    return broadcast_reduce_add_si256_ps(x)[0];
}
static inline __attribute__((always_inline)) double hsum_double_avx(__m256d v) {
    __m128d vlow  = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1); // high 128
            vlow  = _mm_add_pd(vlow, vhigh);     // reduce down to 128

    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return  _mm_cvtsd_f64(_mm_add_sd(vlow, high64));  // reduce to scalar
}
#endif

#ifdef __SSE2__
static inline __attribute__((always_inline)) __m128 broadcast_reduce_add_si128_ps(__m128 x) {
    __m128 m1 = _mm_shuffle_ps(x, x, _MM_SHUFFLE(0,0,3,2));
    __m128 m2 = _mm_add_ps(x, m1);
    __m128 m3 = _mm_shuffle_ps(m2, m2, _MM_SHUFFLE(0,0,0,1));
    return _mm_add_ps(m2, m3);
}
static inline __attribute__((always_inline)) double _mm_reduce_add_pdd(__m128d x) {
    return x[0] + x[1];
}
static inline __attribute__((always_inline)) double _mm_reduce_add_psf(__m128 x) {
    return broadcast_reduce_add_si128_ps(x)[0];
}
#endif

#ifndef __FMA__
#ifdef __AVX2__
#define _mm256_fmadd_ps(a, b, c) (_mm256_add_ps(c, _mm256_mul_ps(a, b)))
#define _mm256_fmadd_pd(a, b, c) (_mm256_add_pd(c, _mm256_mul_pd(a, b)))
#endif
#ifdef __SSE2__
#define _mm_fmadd_ps(a, b, c) (_mm_add_ps(c, _mm_mul_ps(a, b)))
#define _mm_fmadd_pd(a, b, c) (_mm_add_pd(c, _mm_mul_pd(a, b)))
#endif
#endif

#define LIBKL_ALOG_PD_MUL 1.539095918623324e-16
#define LIBKL_ALOG_PD_INC -709.0895657128241
#define LIBKL_ALOG_PS_MUL 8.2629582881927490e-8f
#define LIBKL_ALOG_PS_INC -88.02969186f


#if __AVX512F__

static inline  __attribute__((always_inline)) __m512d _mm512_lkl_alog_pd(__m512d x) {
    return _mm512_fmadd_pd(_mm512_cvtepi64_pd(_mm512_castpd_si512(x)),
                           _mm512_set1_pd(LIBKL_ALOG_PD_MUL),
                           _mm512_set1_pd(LIBKL_ALOG_PD_INC));
}
static inline  __attribute__((always_inline)) __m512 _mm512_lkl_alog_ps(__m512 x) {
    return _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_castps_si512(x)),
                           _mm512_set1_ps(LIBKL_ALOG_PS_MUL),
                           _mm512_set1_ps(LIBKL_ALOG_PS_INC));
}
#endif

#if __AVX2__
static inline __attribute__((always_inline)) __m256 _mm256_kl_abs_ps(__m256 a) {
    return _mm256_max_ps(a, -a);
}
static inline __attribute__((always_inline)) __m256d _mm256_kl_abs_pd(__m256d a) {
    return _mm256_max_pd(a, -a);
}
#ifndef DEFINED_mm256_cvtepi64_pd_manual
static inline __attribute__((always_inline)) __m256d _mm256_cvtepi64_pd_manual(const __m256i v)
// From https://stackoverflow.com/questions/41144668/how-to-efficiently-perform-double-int64-conversions-with-sse-avx/41223013
{
    __m256i magic_i_lo   = _mm256_set1_epi64x(0x4330000000000000);
    __m256i magic_i_hi32 = _mm256_set1_epi64x(0x4530000000000000);
    __m256i magic_i_all  = _mm256_set1_epi64x(0x4530000000100000);
    __m256d magic_d_all  = _mm256_castsi256_pd(magic_i_all);

    __m256i v_lo         = _mm256_blend_epi32(magic_i_lo, v, 0b01010101);
    __m256i v_hi         = _mm256_srli_epi64(v, 32);
            v_hi         = _mm256_xor_si256(v_hi, magic_i_hi32);
    __m256d v_hi_dbl     = _mm256_sub_pd(_mm256_castsi256_pd(v_hi), magic_d_all);
    __m256d result       = _mm256_add_pd(v_hi_dbl, _mm256_castsi256_pd(v_lo));
    return result;
}
#define DEFINED_mm256_cvtepi64_pd_manual
#endif

#ifndef _mm256_cvtepi64_pd
#define _mm256_cvtepi64_pd(x) _mm256_cvtepi64_pd_manual(x)
#endif

static inline  __attribute__((always_inline)) __m256d _mm256_lkl_alog_pd(__m256d x) {
    return _mm256_fmadd_pd(_mm256_cvtepi64_pd(_mm256_castpd_si256(x)),
                           _mm256_set1_pd(LIBKL_ALOG_PD_MUL),
                           _mm256_set1_pd(LIBKL_ALOG_PD_INC));
}
static inline  __attribute__((always_inline)) __m256 _mm256_lkl_alog_ps(__m256 x) {
    return _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_castps_si256(x)),
                           _mm256_set1_ps(LIBKL_ALOG_PS_MUL),
                           _mm256_set1_ps(LIBKL_ALOG_PS_INC));
}
#endif
#if __SSE2__
#ifndef DEFINED_mm_cvtepi64_pd_manual
static inline __attribute__((always_inline)) __m128d _mm_cvtepi64_pd_manual(__m128i x){
    __m128i xH = _mm_srli_epi64(x, 32);
    xH = _mm_or_si128(xH, _mm_castpd_si128(_mm_set1_pd(19342813113834066795298816.)));          //  2^84
    __m128i xL = _mm_blend_epi16(x, _mm_castpd_si128(_mm_set1_pd(0x0010000000000000)), 0xcc);   //  2^52
    __m128d f = _mm_sub_pd(_mm_castsi128_pd(xH), _mm_set1_pd(19342813118337666422669312.));     //  2^84 + 2^52
    return _mm_add_pd(f, _mm_castsi128_pd(xL));
}
#define DEFINED_mm_cvtepi64_pd_manual
#endif

#ifndef _mm_cvtepi64_pd
#define _mm_cvtepi64_pd(x) _mm_cvtepi64_pd_manual(x)
#endif
static inline __attribute__((always_inline)) __m128 _mm_kl_abs_ps(__m128 a) {
    return _mm_max_ps(a, -a);
}
static inline __attribute__((always_inline)) __m128d _mm_kl_abs_pd(__m128d a) {
    return _mm_max_pd(a, -a);
}
static inline  __attribute__((always_inline)) __m128d _mm_lkl_alog_pd(__m128d x) {
    return _mm_fmadd_pd(_mm_cvtepi64_pd(_mm_castpd_si128(x)),
                           _mm_set1_pd(LIBKL_ALOG_PD_MUL),
                           _mm_set1_pd(LIBKL_ALOG_PD_INC));
}
static inline  __attribute__((always_inline)) __m128 _mm_lkl_alog_ps(__m128 x) {
    return _mm_fmadd_ps(_mm_cvtepi32_ps(_mm_castps_si128(x)),
                           _mm_set1_ps(LIBKL_ALOG_PS_MUL),
                           _mm_set1_ps(LIBKL_ALOG_PS_INC));
}
#endif

LIBKL_API double kl_reduce_aligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhi, double rhi) {
    double ret = 0.;
    size_t i = 0;
#if __AVX512F__
#define lhload(i) _mm512_add_pd(_mm512_load_pd(lhs + ((i) * nper)), lhv)
#define rhload(i) _mm512_add_pd(_mm512_load_pd(rhs + ((i) * nper)), rhv)
    __m512d lhv = _mm512_set1_pd(lhi), rhv = _mm512_set1_pd(rhi);
    assert((uint64_t)lhs % 64 == 0);
    const size_t nper = sizeof(__m512) / sizeof(double);
    const size_t nsimd = n / nper;
    const size_t nsimd4 = (nsimd / 4) * 4;
    {
        __m512d sum = _mm512_setzero_pd();
        for(; i < nsimd4; i += 4) {
            __m512d lh0 = lhload(i), lh1 = lhload(i + 1), lh2 = lhload(i + 2), lh3 = lhload(i + 3);
            __m512d rh0 = rhload(i), rh1 = rhload(i + 1), rh2 = rhload(i + 2), rh3 = rhload(i + 3);
            __m512d v0 = _mm512_mul_pd(lh0, Sleef_logd8_u35(_mm512_div_pd(lh0, rh0)));
            __m512d v1 = _mm512_mul_pd(lh1, Sleef_logd8_u35(_mm512_div_pd(lh1, rh1)));
            __m512d v2 = _mm512_mul_pd(lh2, Sleef_logd8_u35(_mm512_div_pd(lh2, rh2)));
            __m512d v3 = _mm512_mul_pd(lh3, Sleef_logd8_u35(_mm512_div_pd(lh3, rh3)));
            sum = _mm512_add_pd(sum,  _mm512_add_pd(_mm512_add_pd(v0, v1), _mm512_add_pd(v2, v3)));
        }
        ret += _mm512_reduce_add_pd(sum);
    }
    for(; i < nsimd; ++i) {
        __m512d lh = lhload(i);
        __m512d logv = Sleef_logd8_u35(_mm512_div_pd(lh, rhload(i)));
        ret += _mm512_reduce_add_pd(_mm512_mul_pd(lh, logv));
    }
#undef lhload
#undef rhload
    i *= nper;
#elif __AVX2__
    assert(((uint64_t)lhs) % 32 == 0);
    const size_t nper = sizeof(__m256) / sizeof(double);
    const size_t nsimd = n / nper;
    __m256d lhv = _mm256_set1_pd(lhi), rhv = _mm256_set1_pd(rhi);
    __m256d sum = _mm256_setzero_pd();
    #pragma GCC unroll 4
    for(; i < nsimd; ++i) {
        __m256d lh = _mm256_add_pd(_mm256_load_pd(lhs + ((i) * nper)), lhv),
                rh = _mm256_add_pd(_mm256_load_pd(rhs + ((i) * nper)), rhv);
        sum = _mm256_add_pd(
            sum,
            _mm256_mul_pd(lh, Sleef_logd4_u35(_mm256_div_pd(lh, rh)))
        );
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
        double lhv = lhs[i] + lhi, rhv = rhs[i] + rhi;
        ret += lhv * logf(lhv / rhv);
    }
    return ret;
}


LIBKL_API double kl_reduce_aligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhi, float rhi) {
    double ret = 0.;
    size_t i = 0;
#if __AVX512F__
    assert(((uint64_t)lhs) % 64 == 0);
    const size_t nper = sizeof(__m512) / sizeof(float);
    const size_t nsimd = n / nper;
    const size_t nsimd4 = (nsimd / 4) * 4;
    for(i = 0; i < nsimd4; i += 4) {
        // Perform 4 vector blocks
        __m512 lh0 = _mm512_add_ps(_mm512_load_ps(lhs + (i * nper)), _mm512_set1_ps(lhi));
        __m512 logd0 = Sleef_logf16_u35(_mm512_div_ps(lh0, _mm512_add_ps(_mm512_load_ps(rhs + (i * nper)), _mm512_set1_ps(rhi))));
        __m512 lh1 = _mm512_add_ps(_mm512_load_ps(lhs + ((i + 1) * nper)), _mm512_set1_ps(lhi));
        __m512 logd1 = Sleef_logf16_u35(_mm512_div_ps(lh1, _mm512_add_ps(_mm512_load_ps(rhs + ((i + 1) * nper)), _mm512_set1_ps(rhi))));
        __m512 lh2 = _mm512_add_ps(_mm512_load_ps(lhs + ((i + 2) * nper)), _mm512_set1_ps(lhi));
        __m512 logd2 = Sleef_logf16_u35(_mm512_div_ps(lh2, _mm512_add_ps(_mm512_load_ps(rhs + ((i + 2) * nper)), _mm512_set1_ps(rhi))));
        __m512 lh3 = _mm512_add_ps(_mm512_load_ps(lhs + ((i + 3) * nper)), _mm512_set1_ps(lhi));
        __m512 logd3 = Sleef_logf16_u35(_mm512_div_ps(lh3, _mm512_add_ps(_mm512_load_ps(rhs + ((i + 3) * nper)), _mm512_set1_ps(rhi))));
        // Accumulate in each pair, then convert to doubles for accumulation
        __m512 s01 = _mm512_fmadd_ps(lh1, logd1, _mm512_mul_ps(lh0, logd0));
        __m512 s23 = _mm512_fmadd_ps(lh2, logd2, _mm512_mul_ps(lh3, logd3));
#ifndef __AVX512DQ__
#  define __get_high_reg(x) _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(x), 1))
#else
#  define __get_high_reg(x) _mm512_extractf32x8_ps(x, 1)
#endif
        __m512d s01d = _mm512_add_pd(_mm512_cvtps_pd(_mm512_castps512_ps256(s01)), _mm512_cvtps_pd(__get_high_reg(s01)));
        __m512d s23d = _mm512_add_pd(_mm512_cvtps_pd(_mm512_castps512_ps256(s23)), _mm512_cvtps_pd(__get_high_reg(s23)));
#undef __get_high_reg
        ret += _mm512_reduce_add_pd(_mm512_add_pd(s01d, s23d));
    }
    __m512 sum = _mm512_setzero_ps();
    for(; i < nsimd; ++i) {
        __m512 lh = _mm512_add_ps(_mm512_load_ps(lhs + (i * nper)), _mm512_set1_ps(lhi));
        __m512 rh = _mm512_add_ps(_mm512_load_ps(rhs + (i * nper)), _mm512_set1_ps(rhi));
        __m512 logv = Sleef_logf16_u35(_mm512_div_ps(lh, rh));
        sum = _mm512_fmadd_ps(lh, logv, sum);
    }
    ret += _mm512_reduce_add_ps(sum);
    i *= nper;
#elif __AVX2__
    assert(((uint64_t)lhs) % 32 == 0);
    const size_t nper = sizeof(__m256) / sizeof(float);
    const size_t nsimd = n / nper;

#define __PI(x) __m256 lh##x = _mm256_add_ps(_mm256_load_ps(lhs + ((i + x) * nper)), _mm256_set1_ps(lhi)); \
               __m256 logd##x = Sleef_logf8_u35(_mm256_div_ps(lh0, _mm256_add_ps(_mm256_load_ps(rhs + ((i + x) * nper)), _mm256_set1_ps(rhi))));
    __m256 sum = _mm256_setzero_ps();
    #pragma GCC unroll 4
    for(; i < nsimd; ++i) {
        __PI(0)
        sum = _mm256_fmadd_ps(lh0, logd0, sum);
    }
#undef __PI
    ret += broadcast_reduce_add_si256_ps(sum)[0];
    i *= nper;
#elif __SSE2__
    assert(((uint64_t)lhs) % 16 == 0);
    const size_t nper = sizeof(__m128) / sizeof(float);
    const size_t nsimd = n / nper;
    __m128 v = _mm_setzero_ps();
    #pragma GCC unroll 4
    for(; i < nsimd; ++i) {
        __m128 lh = _mm_add_ps(_mm_load_ps(lhs + (i * nper)), _mm_set1_ps(lhi));
        __m128 rh = _mm_add_ps(_mm_load_ps(rhs + (i * nper)), _mm_set1_ps(rhi));
        __m128 div = _mm_div_ps(lh, rh);
        __m128 logv = Sleef_logf4_u35(div);
        v = _mm_fmadd_ps(lh, logv, v);
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
    return ret;
}


LIBKL_API double kl_reduce_unaligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhi, double rhi) {
    double ret = 0.;
    size_t i = 0;
#if __AVX512F__
#define lhload(i) _mm512_add_pd(_mm512_loadu_pd(lhs + ((i) * nper)), lhv)
#define rhload(i) _mm512_add_pd(_mm512_loadu_pd(rhs + ((i) * nper)), rhv)
    __m512d lhv = _mm512_set1_pd(lhi), rhv = _mm512_set1_pd(rhi);
    assert(((uint64_t)lhs) % 64 == 0);
    const size_t nper = sizeof(__m512) / sizeof(double);
    const size_t nsimd = n / nper;
    const size_t nsimd4 = (nsimd / 4) * 4;
    {
        __m512d sum = _mm512_setzero_pd();
        for(; i < nsimd4; i += 4) {
            __m512d lh0 = lhload(i), lh1 = lhload(i + 1), lh2 = lhload(i + 2), lh3 = lhload(i + 3);
            __m512d rh0 = rhload(i), rh1 = rhload(i + 1), rh2 = rhload(i + 2), rh3 = rhload(i + 3);
            __m512d v0 = _mm512_mul_pd(lh0, Sleef_logd8_u35(_mm512_div_pd(lh0, rh0)));
            __m512d v1 = _mm512_mul_pd(lh1, Sleef_logd8_u35(_mm512_div_pd(lh1, rh1)));
            __m512d v2 = _mm512_mul_pd(lh2, Sleef_logd8_u35(_mm512_div_pd(lh2, rh2)));
            __m512d v3 = _mm512_mul_pd(lh3, Sleef_logd8_u35(_mm512_div_pd(lh3, rh3)));
            sum = _mm512_add_pd(sum,  _mm512_add_pd(_mm512_add_pd(v0, v1), _mm512_add_pd(v2, v3)));
        }
        ret += _mm512_reduce_add_pd(sum);
    }
    for(; i < nsimd; ++i) {
        __m512d lh = lhload(i);
        __m512d logv = Sleef_logd8_u35(_mm512_div_pd(lh, rhload(i)));
#undef lhload
#undef rhload
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
    return ret;
}

LIBKL_API double kl_reduce_unaligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhi, float rhi) {
    double ret = 0.;
    size_t i = 0;
#if __AVX512F__
    assert(((uint64_t)lhs) % 64 == 0);
    const size_t nper = sizeof(__m512) / sizeof(float);
    const size_t nsimd = n / nper;
    __m512 sum = _mm512_setzero_ps();
    #pragma GCC unroll 4
    for(; i < nsimd; ++i) {
        __m512 lh = _mm512_add_ps(_mm512_loadu_ps(lhs + (i * nper)), _mm512_set1_ps(lhi));
        __m512 rh = _mm512_add_ps(_mm512_loadu_ps(rhs + (i * nper)), _mm512_set1_ps(rhi));
        __m512 div = _mm512_div_ps(lh, rh);
        __m512 logv = Sleef_logf16_u35(div);
        sum = _mm512_fmadd_ps(lh, logv, sum);
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
        sum = _mm256_fmadd_ps(lh, Sleef_logf8_u35(_mm256_div_ps(lh, rh)), sum);
    }
    ret += broadcast_reduce_add_si256_ps(sum)[0];
    i *= nper;
#elif __SSE2__
    assert(((uint64_t)lhs) % 16 == 0);
    const size_t nper = sizeof(__m128) / sizeof(float);
    const size_t nsimd = n / nper;
    __m128 v = _mm_setzero_ps();
    #pragma GCC unroll 4
    for(; i < nsimd; ++i) {
        __m128 lh = _mm_add_ps(_mm_loadu_ps(lhs + (i * nper)), _mm_set1_ps(lhi));
        __m128 rh = _mm_add_ps(_mm_loadu_ps(rhs + (i * nper)), _mm_set1_ps(rhi));
        __m128 div = _mm_div_ps(lh, rh);
        __m128 logv = Sleef_logf4_u35(div);
        v = _mm_fmadd_ps(lh, logv, v);
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
#if 0
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

LIBKL_API double sis_reduce_unaligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhi, float rhi)
{
    double ret = 0.;
    size_t i = 0;
#if __AVX512F__
    assert(((uint64_t)lhs) % 64 == 0);
    const size_t nper = sizeof(__m512) / sizeof(float);
    const size_t nsimd = n / nper;
    __m512 sum = _mm512_set1_ps(0.);
    #pragma GCC unroll 4
    while(i < nsimd) {
        __m512 lh = _mm512_add_ps(_mm512_loadu_ps(lhs + (i * nper)), _mm512_set1_ps(lhi));
        __m512 rh = _mm512_add_ps(_mm512_loadu_ps(rhs + (i * nper)), _mm512_set1_ps(rhi));
        __m512 divf = _mm512_div_ps(lh, rh);
        __m512 divr = _mm512_div_ps(rh, lh);
        sum = _mm512_fmadd_ps(_mm512_add_ps(_mm512_add_ps(divf, divr), _mm512_set1_ps(2.)), _mm512_set1_ps(.25), sum);
        ++i;
    }
    ret += .5 * _mm512_reduce_add_ps(sum);
    i *= nper;
#elif __AVX2__
    assert(((uint64_t)lhs) % 32 == 0);
    const size_t nper = sizeof(__m256) / sizeof(float);
    const size_t nsimd = n / nper;

    __m256 sum = _mm256_set1_ps(0.);
    #pragma GCC unroll 4
    for(; i < nsimd; ++i) {
        __m256 lh = _mm256_add_ps(_mm256_loadu_ps(lhs + (i * nper)), _mm256_set1_ps(lhi));
        __m256 rh = _mm256_add_ps(_mm256_loadu_ps(rhs + (i * nper)), _mm256_set1_ps(rhi));
        __m256 divf = _mm256_div_ps(lh, rh);
        __m256 divr = _mm256_div_ps(rh, lh);
        sum = _mm256_fmadd_ps(_mm256_add_ps(_mm256_add_ps(divf, divr), _mm256_set1_ps(2.)), _mm256_set1_ps(.25), sum);
    }
    ret += .5 * broadcast_reduce_add_si256_ps(sum)[0];
    i *= nper;
#elif __SSE2__
    assert(((uint64_t)lhs) % 16 == 0);
    const size_t nper = sizeof(__m128) / sizeof(float);
    const size_t nsimd = n / nper;
    #pragma GCC unroll 4
    for(; i < nsimd; ++i) {
        __m128 lh = _mm_add_ps(_mm_loadu_ps(lhs + (i * nper)), _mm_set1_ps(lhi));
        __m128 rh = _mm_add_ps(_mm_loadu_ps(rhs + (i * nper)), _mm_set1_ps(rhi));
        __m128 divf = _mm_div_ps(lh, rh);
        __m128 divr = _mm_div_ps(rh, lh);
        __m128 divfr = _mm_mul_ps(_mm_add_ps(_mm_add_ps(divf, divr), _mm_set1_ps(2.)), _mm_set1_ps(.25));
        __m128 res = Sleef_logf4_u35(divfr);
        ret += res[0] + res[1];
    }
    ret *= .5;
    i *= nper;
#endif
    #pragma GCC unroll 8
    for(; i < n; ++i) {
        float lhv = lhs[i] + lhi;
        float rhv = rhs[i] + rhi;
        float divf = lhv / rhv, divr = rhv / lhv;
        ret += logf((divf + divr + 2.) * .25);
    }
    return ret;
}
LIBKL_API double sis_reduce_aligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhi, float rhi)
{
    double ret = 0.;
    size_t i = 0;
#if __AVX512F__
    assert(((uint64_t)lhs) % 64 == 0);
    const size_t nper = sizeof(__m512) / sizeof(float);
    const size_t nsimd = n / nper;
    __m512 sum = _mm512_set1_ps(0.);
    #pragma GCC unroll 4
    while(i < nsimd) {
        __m512 lh = _mm512_add_ps(_mm512_load_ps(lhs + (i * nper)), _mm512_set1_ps(lhi));
        __m512 rh = _mm512_add_ps(_mm512_load_ps(rhs + (i * nper)), _mm512_set1_ps(rhi));
        __m512 divf = _mm512_div_ps(lh, rh);
        __m512 divr = _mm512_div_ps(rh, lh);
        sum = _mm512_fmadd_ps(_mm512_add_ps(_mm512_add_ps(divf, divr), _mm512_set1_ps(2.)), _mm512_set1_ps(.25), sum);
        ++i;
    }
    ret += .5 * _mm512_reduce_add_ps(sum);
    i *= nper;
#elif __AVX2__
    assert(((uint64_t)lhs) % 32 == 0);
    const size_t nper = sizeof(__m256) / sizeof(float);
    const size_t nsimd = n / nper;

    __m256 sum = _mm256_set1_ps(0.);
    #pragma GCC unroll 4
    for(; i < nsimd; ++i) {
        __m256 lh = _mm256_add_ps(_mm256_load_ps(lhs + (i * nper)), _mm256_set1_ps(lhi));
        __m256 rh = _mm256_add_ps(_mm256_load_ps(rhs + (i * nper)), _mm256_set1_ps(rhi));
        __m256 divf = _mm256_div_ps(lh, rh);
        __m256 divr = _mm256_div_ps(rh, lh);
        __m256 divfr = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(divf, divr), _mm256_set1_ps(2.)), _mm256_set1_ps(.25));
        sum = _mm256_add_ps(sum, Sleef_logf8_u35(divfr));
    }
    ret += .5 * broadcast_reduce_add_si256_ps(sum)[0];
    i *= nper;
#elif __SSE2__
    assert(((uint64_t)lhs) % 16 == 0);
    const size_t nper = sizeof(__m128) / sizeof(float);
    const size_t nsimd = n / nper;
    #pragma GCC unroll 4
    for(; i < nsimd; ++i) {
        __m128 lh = _mm_add_ps(_mm_load_ps(lhs + (i * nper)), _mm_set1_ps(lhi));
        __m128 rh = _mm_add_ps(_mm_load_ps(rhs + (i * nper)), _mm_set1_ps(rhi));
        __m128 divf = _mm_div_ps(lh, rh);
        __m128 divr = _mm_div_ps(rh, lh);
        __m128 divfr = _mm_mul_ps(_mm_add_ps(_mm_add_ps(divf, divr), _mm_set1_ps(2.)), _mm_set1_ps(.25));
        __m128 res = Sleef_logf4_u35(divfr);
        ret += res[0] + res[1];
    }
    ret *= .5;
    i *= nper;
#endif
    #pragma GCC unroll 8
    for(; i < n; ++i) {
        float lhv = lhs[i] + lhi;
        float rhv = rhs[i] + rhi;
        float divf = lhv / rhv, divr = rhv / lhv;
        ret += logf((divf + divr + 2.) * .25);
    }
    return ret;
}

LIBKL_API double sis_reduce_unaligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhi, double rhi)
{
    double ret = 0.;
    size_t i = 0;
#if __AVX512F__
    assert(((uint64_t)lhs) % 64 == 0);
    const size_t nper = sizeof(__m512d) / sizeof(double);
    const size_t nsimd = n / nper;
    __m512d sum = _mm512_set1_pd(0.);
    // Contribution = log((x / 4y) + (y / 4x) + .5)
    // Contribution = log(((x / y) + (y / x) + 2) / 4)
    #pragma GCC unroll 4
    while(i < nsimd) {
        __m512d lh = _mm512_add_pd(_mm512_loadu_pd(lhs + (i * nper)), _mm512_set1_pd(lhi));
        __m512d rh = _mm512_add_pd(_mm512_loadu_pd(rhs + (i * nper)), _mm512_set1_pd(rhi));
        __m512d divf = _mm512_div_pd(lh, rh);
        __m512d divr = _mm512_div_pd(rh, lh);
        sum = _mm512_fmadd_pd(_mm512_add_pd(_mm512_add_pd(divf, divr), _mm512_set1_pd(2.)), _mm512_set1_pd(.25), sum);
        ++i;
    }
    ret += .5 * _mm512_reduce_add_pd(sum);
    i *= nper;
#elif __AVX2__
    assert(((uint64_t)lhs) % 32 == 0);
    const size_t nper = sizeof(__m256d) / sizeof(double);
    const size_t nsimd = n / nper;

    __m256d sum = _mm256_set1_pd(0.);
    for(; i < nsimd; ++i) {
        __m256d lh = _mm256_add_pd(_mm256_loadu_pd(lhs + (i * nper)), _mm256_set1_pd(lhi));
        __m256d rh = _mm256_add_pd(_mm256_loadu_pd(rhs + (i * nper)), _mm256_set1_pd(rhi));
        __m256d divf = _mm256_div_pd(lh, rh);
        __m256d divr = _mm256_div_pd(rh, lh);
        sum = _mm256_fmadd_pd(_mm256_add_pd(_mm256_add_pd(divf, divr), _mm256_set1_pd(2.)), _mm256_set1_pd(.25), sum);
    }
    ret += .5 * hsum_double_avx(sum);
    i *= nper;
#elif __SSE2__
    assert(((uint64_t)lhs) % 16 == 0);
    const size_t nper = sizeof(__m128) / sizeof(double);
    const size_t nsimd = n / nper;
    #pragma GCC unroll 4
    for(; i < nsimd; ++i) {
        __m128d lh = _mm_add_pd(_mm_loadu_pd(lhs + (i * nper)), _mm_set1_pd(lhi));
        __m128d rh = _mm_add_pd(_mm_loadu_pd(rhs + (i * nper)), _mm_set1_pd(rhi));
        __m128d divf = _mm_div_pd(lh, rh);
        __m128d divr = _mm_div_pd(rh, lh);
        __m128d divfr = _mm_mul_pd(_mm_add_pd(divr, _mm_add_pd(divf, _mm_set1_pd(2.))), _mm_set1_pd(.25));
        __m128d res = Sleef_logd2_u35(divfr);
        ret += res[0] + res[1];
    }
    i *= nper;
#endif
    #pragma GCC unroll 8
    for(; i < n; ++i) {
        double lhv = lhs[i] + lhi;
        double rhv = rhs[i] + rhi;
        double divf = lhv / rhv, divr = rhv / lhv;
        ret += log((divf + divr + 2.) * .25);
    }
    return ret;
}
LIBKL_API double sis_reduce_aligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhi, double rhi)
{
    double ret = 0.;
    size_t i = 0;
#if __AVX512F__
    assert(((uint64_t)lhs) % 64 == 0);
    const size_t nper = sizeof(__m512d) / sizeof(double);
    const size_t nsimd = n / nper;
    __m512d sum = _mm512_set1_pd(0.);
    // Contribution = log((x / 4y) + (y / 4x) + .5)
    // Contribution = log(((x / y) + (y / x) + 2) / 4)
    #pragma GCC unroll 4
    while(i < nsimd) {
        __m512d lh = _mm512_add_pd(_mm512_load_pd(lhs + (i * nper)), _mm512_set1_pd(lhi));
        __m512d rh = _mm512_add_pd(_mm512_load_pd(rhs + (i * nper)), _mm512_set1_pd(rhi));
        __m512d divf = _mm512_div_pd(lh, rh);
        __m512d divr = _mm512_div_pd(rh, lh);
        __m512d tmp = Sleef_logd8_u35(_mm512_mul_pd(_mm512_add_pd(_mm512_add_pd(divf, divr), _mm512_set1_pd(2.)), _mm512_set1_pd(.25)));
        sum = _mm512_add_pd(tmp, sum);
        ++i;
    }
    ret += .5 * _mm512_reduce_add_pd(sum);
    i *= nper;
#elif __AVX2__
    assert(((uint64_t)lhs) % 32 == 0);
    const size_t nper = sizeof(__m256d) / sizeof(double);
    const size_t nsimd = n / nper;

    __m256d sum = _mm256_set1_pd(0.);
    #pragma GCC unroll 4
    for(; i < nsimd; ++i) {
        __m256d lh = _mm256_add_pd(_mm256_load_pd(lhs + (i * nper)), _mm256_set1_pd(lhi));
        __m256d rh = _mm256_add_pd(_mm256_load_pd(rhs + (i * nper)), _mm256_set1_pd(rhi));
        __m256d divf = _mm256_div_pd(lh, rh);
        __m256d divr = _mm256_div_pd(rh, lh);
        __m256d tmp = Sleef_logd4_u35(_mm256_mul_pd(_mm256_add_pd(_mm256_add_pd(divf, divr), _mm256_set1_pd(2.)), _mm256_set1_pd(.25)));
        sum = _mm256_add_pd(tmp, sum);
    }
    ret += .5 * hsum_double_avx(sum);
    i *= nper;
#elif __SSE2__
    assert(((uint64_t)lhs) % 16 == 0);
    const size_t nper = sizeof(__m128) / sizeof(double);
    const size_t nsimd = n / nper;
    #pragma GCC unroll 4
    for(; i < nsimd; ++i) {
        __m128d lh = _mm_add_pd(_mm_load_pd(lhs + (i * nper)), _mm_set1_pd(lhi));
        __m128d rh = _mm_add_pd(_mm_load_pd(rhs + (i * nper)), _mm_set1_pd(rhi));
        __m128d divf = _mm_div_pd(lh, rh);
        __m128d divr = _mm_div_pd(rh, lh);
        __m128d divfr = _mm_mul_pd(_mm_add_pd(divr, _mm_add_pd(divf, _mm_set1_pd(2.))), _mm_set1_pd(.25));
        __m128d res = Sleef_logd2_u35(divfr);
        ret += res[0] + res[1];
    }
    i *= nper;
#endif
    #pragma GCC unroll 8
    for(; i < n; ++i) {
        double lhv = lhs[i] + lhi;
        double rhv = rhs[i] + rhi;
        double divf = lhv / rhv, divr = rhv / lhv;
        ret += log((divf + divr + 2.) * .25);
    }
    return ret;
}

LIBKL_API double is_reduce_aligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhi, double rhi)
{
    double ret = 0.;
    size_t i = 0;
#if __AVX512F__
    assert(((uint64_t)lhs) % 64 == 0);
    const size_t nper = sizeof(__m512d) / sizeof(double);
    const size_t nsimd = n / nper;
    const size_t nsimd4 = (nsimd / 4) * 4;

    for(; i < nsimd4; i += 4) {
        __m512d lh = _mm512_add_pd(_mm512_load_pd(lhs + (i * nper)), _mm512_set1_pd(lhi));
        __m512d rh = _mm512_add_pd(_mm512_load_pd(rhs + (i * nper)), _mm512_set1_pd(rhi));
        __m512d div0 = _mm512_div_pd(lh, rh);
        __m512d sum0 = _mm512_sub_pd(div0, Sleef_logd8_u35(div0));
        __m512d lh1 = _mm512_add_pd(_mm512_load_pd(lhs + ((i + 1) * nper)), _mm512_set1_pd(lhi));
        __m512d rh1 = _mm512_add_pd(_mm512_load_pd(rhs + ((i + 1) * nper)), _mm512_set1_pd(rhi));
        __m512d div1 = _mm512_div_pd(lh1, rh1);
        __m512d sum1 = _mm512_sub_pd(div1, Sleef_logd8_u35(div1));
        __m512d lh2 = _mm512_add_pd(_mm512_load_pd(lhs + ((i + 2) * nper)), _mm512_set1_pd(lhi));
        __m512d rh2 = _mm512_add_pd(_mm512_load_pd(rhs + ((i + 2) * nper)), _mm512_set1_pd(rhi));
        __m512d div2 = _mm512_div_pd(lh2, rh2);
        __m512d sum2 = _mm512_sub_pd(div2, Sleef_logd8_u35(div2));
        __m512d lh3 = _mm512_add_pd(_mm512_load_pd(lhs + ((i + 3) * nper)), _mm512_set1_pd(lhi));
        __m512d rh3 = _mm512_add_pd(_mm512_load_pd(rhs + ((i + 3) * nper)), _mm512_set1_pd(rhi));
        __m512d div3 = _mm512_div_pd(lh3, rh3);
        __m512d sum3 = _mm512_sub_pd(div3, Sleef_logd8_u35(div3));
        ret += _mm512_reduce_add_pd(_mm512_add_pd(_mm512_add_pd(sum0, sum1), _mm512_add_pd(sum2, sum3)));
    }
    while(i < nsimd) {
        __m512d lh = _mm512_add_pd(_mm512_load_pd(lhs + (i * nper)), _mm512_set1_pd(lhi));
        __m512d rh = _mm512_add_pd(_mm512_load_pd(rhs + (i * nper)), _mm512_set1_pd(rhi));
        __m512d div = _mm512_div_pd(lh, rh);
        ret += _mm512_reduce_add_pd(_mm512_sub_pd(div, Sleef_logd8_u35(div)));
        ++i;
    }
    i *= nper;
#elif __AVX2__
    assert(((uint64_t)lhs) % 32 == 0);
    const size_t nper = sizeof(__m256d) / sizeof(double);
    const size_t nsimd = n / nper;
    const size_t nsimd4 = (nsimd / 4) * 4;

#define __PERF(j)\
        __m256d lhl##j = _mm256_load_pd(lhs + ((i + (j))  * nper));\
        __m256d rhl##j = _mm256_load_pd(rhs + ((i + (j))  * nper));\
        __m256d lh##j = _mm256_add_pd(lhl##j, _mm256_set1_pd(lhi));\
        __m256d rh##j = _mm256_add_pd(rhl##j, _mm256_set1_pd(rhi));\
        __m256d div##j = _mm256_div_pd(lh##j, rh##j);\
        __m256d res##j = _mm256_sub_pd(div##j, Sleef_logd4_u35(div##j));

    for(; i < nsimd4; i += 4) {
        __PERF(0) __PERF(1) __PERF(2) __PERF(3)
        double inc = hsum_double_avx(_mm256_add_pd(_mm256_add_pd(res0, res1), _mm256_add_pd(res2, res3)));
        ret += inc;
    }
    for(; i < nsimd; ++i) {
        __PERF(0)
#undef __PERF
        ret += hsum_double_avx(res0);
    }
    i *= nper;
#elif __SSE2__
    assert(((uint64_t)lhs) % 16 == 0);
    const size_t nper = sizeof(__m128) / sizeof(double);
    const size_t nsimd = n / nper;
    #pragma GCC unroll 4
    for(; i < nsimd; ++i) {
        __m128d lh = _mm_add_pd(_mm_load_pd(lhs + (i * nper)), _mm_set1_pd(lhi));
        __m128d rh = _mm_add_pd(_mm_load_pd(rhs + (i * nper)), _mm_set1_pd(rhi));
        __m128d div = _mm_div_pd(lh, rh);
        __m128d res = _mm_sub_pd(div, Sleef_logd2_u35(div));
        ret += res[0] + res[1];
    }
    i *= nper;
#endif
    #pragma GCC unroll 8
    for(; i < n; ++i) {
        double lhv = lhs[i] + lhi;
        double rhv = rhs[i] + rhi;
        double div = lhv / rhv;
        ret += div - log(div);
    }
    return ret;
}
LIBKL_API double is_reduce_unaligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhi, double rhi)
{
    double ret = 0.;
    size_t i = 0;
#if __AVX512F__
    assert(((uint64_t)lhs) % 64 == 0);
    const size_t nper = sizeof(__m512d) / sizeof(double);
    const size_t nsimd = n / nper;
    const size_t nsimd4 = (nsimd / 4) * 4;

    for(; i < nsimd4; i += 4) {
        __m512d lh = _mm512_add_pd(_mm512_loadu_pd(lhs + (i * nper)), _mm512_set1_pd(lhi));
        __m512d rh = _mm512_add_pd(_mm512_loadu_pd(rhs + (i * nper)), _mm512_set1_pd(rhi));
        __m512d div0 = _mm512_div_pd(lh, rh);
        __m512d sum0 = _mm512_sub_pd(div0, Sleef_logd8_u35(div0));
        __m512d lh1 = _mm512_add_pd(_mm512_loadu_pd(lhs + ((i + 1) * nper)), _mm512_set1_pd(lhi));
        __m512d rh1 = _mm512_add_pd(_mm512_loadu_pd(rhs + ((i + 1) * nper)), _mm512_set1_pd(rhi));
        __m512d div1 = _mm512_div_pd(lh1, rh1);
        __m512d sum1 = _mm512_sub_pd(div1, Sleef_logd8_u35(div1));
        __m512d lh2 = _mm512_add_pd(_mm512_loadu_pd(lhs + ((i + 2) * nper)), _mm512_set1_pd(lhi));
        __m512d rh2 = _mm512_add_pd(_mm512_loadu_pd(rhs + ((i + 2) * nper)), _mm512_set1_pd(rhi));
        __m512d div2 = _mm512_div_pd(lh2, rh2);
        __m512d sum2 = _mm512_sub_pd(div2, Sleef_logd8_u35(div2));
        __m512d lh3 = _mm512_add_pd(_mm512_loadu_pd(lhs + ((i + 3) * nper)), _mm512_set1_pd(lhi));
        __m512d rh3 = _mm512_add_pd(_mm512_loadu_pd(rhs + ((i + 3) * nper)), _mm512_set1_pd(rhi));
        __m512d div3 = _mm512_div_pd(lh3, rh3);
        __m512d sum3 = _mm512_sub_pd(div3, Sleef_logd8_u35(div3));
        ret += _mm512_reduce_add_pd(_mm512_add_pd(_mm512_add_pd(sum0, sum1), _mm512_add_pd(sum2, sum3)));
    }
    while(i < nsimd) {
        __m512d lh = _mm512_add_pd(_mm512_loadu_pd(lhs + (i * nper)), _mm512_set1_pd(lhi));
        __m512d rh = _mm512_add_pd(_mm512_loadu_pd(rhs + (i * nper)), _mm512_set1_pd(rhi));
        __m512d div = _mm512_div_pd(lh, rh);
        ret += _mm512_reduce_add_pd(_mm512_sub_pd(div, Sleef_logd8_u35(div)));
        ++i;
    }
    i *= nper;
#elif __AVX2__
    assert(((uint64_t)lhs) % 32 == 0);
    const size_t nper = sizeof(__m256d) / sizeof(double);
    const size_t nsimd = n / nper;
    const size_t nsimd4 = (nsimd / 4) * 4;

    for(; i < nsimd4; i += 4) {
#define __PERF(j)\
        __m256d lh##j = _mm256_add_pd(_mm256_loadu_pd(lhs + ((i + (j))  * nper)), _mm256_set1_pd(lhi));\
        __m256d rh##j = _mm256_add_pd(_mm256_loadu_pd(rhs + ((i + (j))  * nper)), _mm256_set1_pd(rhi));\
        __m256d div##j = _mm256_div_pd(lh##j, rh##j);\
        __m256d res##j = _mm256_sub_pd(div##j, Sleef_logd4_u35(div##j));\

        __PERF(0) __PERF(1) __PERF(2) __PERF(3)
        ret += hsum_double_avx(_mm256_add_pd(_mm256_add_pd(res0, res1), _mm256_add_pd(res2, res3)));
    }
    for(; i < nsimd; ++i) {
        __PERF(0)
#undef __PERF
        ret += hsum_double_avx(res0);
    }
    i *= nper;
#elif __SSE2__
    assert(((uint64_t)lhs) % 16 == 0);
    const size_t nper = sizeof(__m128) / sizeof(double);
    const size_t nsimd = n / nper;
    #pragma GCC unroll 4
    for(; i < nsimd; ++i) {
        __m128d lh = _mm_add_pd(_mm_loadu_pd(lhs + (i * nper)), _mm_set1_pd(lhi));
        __m128d rh = _mm_add_pd(_mm_loadu_pd(rhs + (i * nper)), _mm_set1_pd(rhi));
        __m128d div = _mm_div_pd(lh, rh);
        __m128d res = _mm_sub_pd(div, Sleef_logd2_u35(div));
        ret += res[0] + res[1];
    }
    i *= nper;
#endif
    #pragma GCC unroll 8
    for(; i < n; ++i) {
        double lhv = lhs[i] + lhi;
        double rhv = rhs[i] + rhi;
        double div = lhv / rhv;
        ret += div - logf(div);
    }
    return ret;
}

LIBKL_API double is_reduce_aligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhi, float rhi)
{
    double ret = 0.;
    size_t i = 0;
#if __AVX512F__
    assert(((uint64_t)lhs) % 64 == 0);
    const size_t nper = sizeof(__m512) / sizeof(float);
    const size_t nsimd = n / nper;
    const size_t nsimd4 = (nsimd / 4) * 4;

    for(; i < nsimd4; i += 4) {
        __m512 lh = _mm512_add_ps(_mm512_load_ps(lhs + (i * nper)), _mm512_set1_ps(lhi));
        __m512 rh = _mm512_add_ps(_mm512_load_ps(rhs + (i * nper)), _mm512_set1_ps(rhi));
        __m512 div0 = _mm512_div_ps(lh, rh);
        __m512 sum0 = _mm512_sub_ps(div0, Sleef_logf16_u35(div0));
        __m512 lh1 = _mm512_add_ps(_mm512_load_ps(lhs + ((i + 1) * nper)), _mm512_set1_ps(lhi));
        __m512 rh1 = _mm512_add_ps(_mm512_load_ps(rhs + ((i + 1) * nper)), _mm512_set1_ps(rhi));
        __m512 div1 = _mm512_div_ps(lh1, rh1);
        __m512 sum1 = _mm512_sub_ps(div1, Sleef_logf16_u35(div1));
        __m512 lh2 = _mm512_add_ps(_mm512_load_ps(lhs + ((i + 2) * nper)), _mm512_set1_ps(lhi));
        __m512 rh2 = _mm512_add_ps(_mm512_load_ps(rhs + ((i + 2) * nper)), _mm512_set1_ps(rhi));
        __m512 div2 = _mm512_div_ps(lh2, rh2);
        __m512 sum2 = _mm512_sub_ps(div2, Sleef_logf16_u35(div2));
        __m512 lh3 = _mm512_add_ps(_mm512_load_ps(lhs + ((i + 3) * nper)), _mm512_set1_ps(lhi));
        __m512 rh3 = _mm512_add_ps(_mm512_load_ps(rhs + ((i + 3) * nper)), _mm512_set1_ps(rhi));
        __m512 div3 = _mm512_div_ps(lh3, rh3);
        __m512 sum3 = _mm512_sub_ps(div3, Sleef_logf16_u35(div3));
        ret += _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3)));
    }
    while(i < nsimd) {
        __m512 lh = _mm512_add_ps(_mm512_load_ps(lhs + (i * nper)), _mm512_set1_ps(lhi));
        __m512 rh = _mm512_add_ps(_mm512_load_ps(rhs + (i * nper)), _mm512_set1_ps(rhi));
        __m512 div = _mm512_div_ps(lh, rh);
        ret += _mm512_reduce_add_ps(_mm512_sub_ps(div, Sleef_logf16_u35(div)));
        ++i;
    }
    i *= nper;
#elif __AVX2__
    assert(((uint64_t)lhs) % 32 == 0);
    const size_t nper = sizeof(__m256) / sizeof(float);
    const size_t nsimd = n / nper;
    const size_t nsimd4 = (nsimd / 4) * 4;

    for(; i < nsimd4; i += 4) {
#define __PERF(j)\
        __m256 lh##j = _mm256_add_ps(_mm256_load_ps(lhs + ((i + (j))  * nper)), _mm256_set1_ps(lhi));\
        __m256 rh##j = _mm256_add_ps(_mm256_load_ps(rhs + ((i + (j))  * nper)), _mm256_set1_ps(rhi));\
        __m256 div##j = _mm256_div_ps(lh##j, rh##j);\
        __m256 res##j = _mm256_sub_ps(div##j, Sleef_logf8_u35(div##j));\

        __PERF(0) __PERF(1) __PERF(2) __PERF(3)
        ret += broadcast_reduce_add_si256_ps(_mm256_add_ps(_mm256_add_ps(res0, res1), _mm256_add_ps(res2, res3)))[0];
    }
    for(; i < nsimd; ++i) {
        __PERF(0)
#undef __PERF
        ret += broadcast_reduce_add_si256_ps(res0)[0];
    }
    i *= nper;
#elif __SSE2__
    assert(((uint64_t)lhs) % 16 == 0);
    const size_t nper = sizeof(__m128) / sizeof(float);
    const size_t nsimd = n / nper;
    #pragma GCC unroll 4
    for(; i < nsimd; ++i) {
        __m128 lh = _mm_add_ps(_mm_load_ps(lhs + (i * nper)), _mm_set1_ps(lhi));
        __m128 rh = _mm_add_ps(_mm_load_ps(rhs + (i * nper)), _mm_set1_ps(rhi));
        __m128 div = _mm_div_ps(lh, rh);
        __m128 res = _mm_sub_ps(div, Sleef_logf4_u35(div));
        ret += broadcast_reduce_add_si128_ps(res)[0];
    }
    i *= nper;
#endif
    #pragma GCC unroll 8
    for(; i < n; ++i) {
        float lhv = lhs[i] + lhi;
        float rhv = rhs[i] + rhi;
        float div = lhv / rhv;
        ret += div - logf(div);
    }
#if 0
    double oret = 0.;
    for(size_t j = 0; j < n; ++j) {
        float lhv = lhs[j] + lhi;
        float rhv = rhs[j] + rhi;
        float div = lhv / rhv;
        oret += div - logf(div);
    }
    assert(fabs(oret - ret) < 1e-5 || !fprintf(stderr, "ret: %g. oret: %g\n", ret, oret));
#endif
    return ret;
}
LIBKL_API double is_reduce_unaligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhi, float rhi)
{
    double ret = 0.;
    size_t i = 0;
#if __AVX512F__
    assert(((uint64_t)lhs) % 64 == 0);
    const size_t nper = sizeof(__m512) / sizeof(float);
    const size_t nsimd = n / nper;
    const size_t nsimd4 = (nsimd / 4) * 4;

    for(; i < nsimd4; i += 4) {
        __m512 lh0 = _mm512_add_ps(_mm512_loadu_ps(lhs + (i * nper)), _mm512_set1_ps(lhi));
        __m512 rh0 = _mm512_add_ps(_mm512_loadu_ps(rhs + (i * nper)), _mm512_set1_ps(rhi));
        __m512 div0 = _mm512_div_ps(lh0, rh0);
        __m512 sum0 = _mm512_sub_ps(div0, Sleef_logf16_u35(div0));
        __m512 lh1 = _mm512_add_ps(_mm512_loadu_ps(lhs + ((i + 1) * nper)), _mm512_set1_ps(lhi));
        __m512 rh1 = _mm512_add_ps(_mm512_loadu_ps(rhs + ((i + 1) * nper)), _mm512_set1_ps(rhi));
        __m512 div1 = _mm512_div_ps(lh1, rh1);
        __m512 sum1 = _mm512_sub_ps(div1, Sleef_logf16_u35(div1));
        __m512 lh2 = _mm512_add_ps(_mm512_loadu_ps(lhs + ((i + 2) * nper)), _mm512_set1_ps(lhi));
        __m512 rh2 = _mm512_add_ps(_mm512_loadu_ps(rhs + ((i + 2) * nper)), _mm512_set1_ps(rhi));
        __m512 div2 = _mm512_div_ps(lh2, rh2);
        __m512 sum2 = _mm512_sub_ps(div2, Sleef_logf16_u35(div2));
        __m512 lh3 = _mm512_add_ps(_mm512_loadu_ps(lhs + ((i + 3) * nper)), _mm512_set1_ps(lhi));
        __m512 rh3 = _mm512_add_ps(_mm512_loadu_ps(rhs + ((i + 3) * nper)), _mm512_set1_ps(rhi));
        __m512 div3 = _mm512_div_ps(lh3, rh3);
        __m512 sum3 = _mm512_sub_ps(div3, Sleef_logf16_u35(div3));
        ret += _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3)));
    }
    while(i < nsimd) {
        __m512 lh = _mm512_add_ps(_mm512_loadu_ps(lhs + (i * nper)), _mm512_set1_ps(lhi));
        __m512 rh = _mm512_add_ps(_mm512_loadu_ps(rhs + (i * nper)), _mm512_set1_ps(rhi));
        __m512 div = _mm512_div_ps(lh, rh);
        ret += _mm512_reduce_add_ps(_mm512_sub_ps(div, Sleef_logf16_u35(div)));
        ++i;
    }
    i *= nper;
#elif __AVX2__
    assert(((uint64_t)lhs) % 32 == 0);
    const size_t nper = sizeof(__m256) / sizeof(float);
    const size_t nsimd = n / nper;
    const size_t nsimd4 = (nsimd / 4) * 4;

    for(; i < nsimd4; i += 4) {
#define __PERF(j)\
        __m256 lh##j = _mm256_add_ps(_mm256_loadu_ps(lhs + ((i + (j))  * nper)), _mm256_set1_ps(lhi));\
        __m256 rh##j = _mm256_add_ps(_mm256_loadu_ps(rhs + ((i + (j))  * nper)), _mm256_set1_ps(rhi));\
        __m256 div##j = _mm256_div_ps(lh##j, rh##j);\
        __m256 res##j = _mm256_sub_ps(div##j, Sleef_logf8_u35(div##j));\

        __PERF(0) __PERF(1) __PERF(2) __PERF(3)
        ret += broadcast_reduce_add_si256_ps(_mm256_add_ps(_mm256_add_ps(res0, res1), _mm256_add_ps(res2, res3)))[0];
    }
    for(; i < nsimd; ++i) {
        __PERF(0)
#undef __PERF
        ret += broadcast_reduce_add_si256_ps(res0)[0];
    }
    i *= nper;
#elif __SSE2__
    assert(((uint64_t)lhs) % 16 == 0);
    const size_t nper = sizeof(__m128) / sizeof(float);
    const size_t nsimd = n / nper;
    #pragma GCC unroll 4
    for(; i < nsimd; ++i) {
        __m128 lh = _mm_add_ps(_mm_loadu_ps(lhs + (i * nper)), _mm_set1_ps(lhi));
        __m128 rh = _mm_add_ps(_mm_loadu_ps(rhs + (i * nper)), _mm_set1_ps(rhi));
        __m128 div = _mm_div_ps(lh, rh);
        __m128 res = _mm_sub_ps(div, Sleef_logf4_u35(div));
        ret += broadcast_reduce_add_si128_ps(res)[0];
    }
    i *= nper;
#endif
    #pragma GCC unroll 8
    for(; i < n; ++i) {
        float lhv = lhs[i] + lhi;
        float rhv = rhs[i] + rhi;
        float div = lhv / rhv;
        ret += div - logf(div);
    }
    return ret;
}

LIBKL_API double llr_reduce_aligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, size_t n, double lambda, float lhinc, float rhinc)
{
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
        __m512 mv = _mm512_fmadd_ps(vlambda, lhl,  _mm512_mul_ps(vm1l, rhl));
        lhsum = _mm512_fmadd_ps(lhl, Sleef_logf16_u35(_mm512_div_ps(lhl, mv)), lhsum);
        rhsum = _mm512_fmadd_ps(rhl, Sleef_logf16_u35(_mm512_div_ps(rhl, mv)), rhsum);
    }
    ret += lambda * _mm512_reduce_add_ps(lhsum) + m1l * _mm512_reduce_add_ps(rhsum);
    for(i *= nperel;i < n; ++i) {
        float xv = lhs[i] + lhinc;
        float yv = rhs[i] + rhinc;
        float mnv = (xv * lambda + yv * m1l);
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
#if LOG3
        __m256 lmv = Sleef_logf8_u35(_mm256_add_ps(_mm256_mul_ps(vlambda, lhsa), _mm256_mul_ps(vm1l, rhsa)));
        lhsum = _mm256_add_ps(lhsum, _mm256_mul_ps(lhsa, _mm256_sub_ps(Sleef_logf8_u35(lhsa), lmv)));
        rhsum = _mm256_add_ps(rhsum, _mm256_mul_ps(rhsa, _mm256_sub_ps(Sleef_logf8_u35(rhsa), lmv)));
#else
        __m256 mv = _mm256_fmadd_ps(vlambda, lhsa,  _mm256_mul_ps(vm1l, rhsa));
        lhsum = _mm256_fmadd_ps(lhsa, Sleef_logf8_u35(_mm256_div_ps(lhsa, mv)), lhsum);
        rhsum = _mm256_fmadd_ps(rhsa, Sleef_logf8_u35(_mm256_div_ps(rhsa, mv)), rhsum);
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
    //
    __m128 vlambda = _mm_set1_ps(lambda), vm1l = _mm_set1_ps(m1l);
    #pragma GCC unroll 4
    for(; i < nsimd; ++i) {
        __m128 lhsa = _mm_add_ps(_mm_load_ps(&lhs[i * nperel]), _mm_set1_ps(lhinc));
        __m128 rhsa = _mm_add_ps(_mm_load_ps(&rhs[i * nperel]), _mm_set1_ps(rhinc));
        __m128 mv = _mm_add_ps(_mm_mul_ps(vlambda, lhsa), _mm_mul_ps(vm1l, rhsa));
        lhsum = _mm_fmadd_ps(lhsa, Sleef_logf4_u35(_mm_div_ps(lhsa, mv)), lhsum);
        rhsum = _mm_fmadd_ps(rhsa, Sleef_logf4_u35(_mm_div_ps(rhsa, mv)), rhsum);
    }
    ret += lambda * broadcast_reduce_add_si128_ps(lhsum)[0] + m1l * broadcast_reduce_add_si128_ps(rhsum)[0];
    for(i *= nperel;i < n; ++i) {
        float lhv = lhs[i] + lhinc;
        float rhv = rhs[i] + rhinc;
        float miv = lhv * lambda + rhv * m1l;
        ret += lambda * lhv * logf(lhv / miv)
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

LIBKL_API double jsd_reduce_aligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, size_t n, float lhinc, float rhinc)
{
    double ret = 0.;
    size_t i = 0;
#if __AVX512F__
    const size_t nperel = sizeof(__m512) / sizeof(float);
    __m512 sum = _mm512_setzero_ps();
    size_t nsimd = n / nperel;
    //
    __m512 lhv = _mm512_set1_ps(lhinc);
    __m512 rhv = _mm512_set1_ps(rhinc);
    #pragma GCC unroll 4
    for(i = 0; i < nsimd; ++i) {
        __m512 lhl = _mm512_add_ps(_mm512_load_ps(&lhs[i * nperel]), lhv);
        __m512 rhl = _mm512_add_ps(_mm512_load_ps(&rhs[i * nperel]), rhv);
        __m512 mv = _mm512_mul_ps(_mm512_set1_ps(.5), _mm512_add_ps(lhl, rhl));
        sum = _mm512_fmadd_ps(lhl, Sleef_logf16_u35(_mm512_div_ps(lhl, mv)), sum);
        sum = _mm512_fmadd_ps(rhl, Sleef_logf16_u35(_mm512_div_ps(rhl, mv)), sum);
    }
    ret += .5f * (_mm512_reduce_add_ps(sum));
    for(i *= nperel;i < n; ++i) {
        float xv = lhs[i] + lhinc;
        float yv = rhs[i] + rhinc;
        float mnv = (xv + yv) * .5f;
        ret += .5f * (xv * log(xv / mnv) + yv * log(yv / mnv));
    }
#elif __AVX2__
    const size_t nperel = sizeof(__m256) / sizeof(float);
    __m256 sum = _mm256_setzero_ps();
    size_t nsimd = n / nperel;
    __m256 lhv = _mm256_set1_ps(lhinc);
    __m256 rhv = _mm256_set1_ps(rhinc);
    #pragma GCC unroll 4
    for(i = 0; i < nsimd; ++i) {
        __m256 lhl = _mm256_add_ps(_mm256_load_ps(&lhs[i * nperel]), lhv);
        __m256 rhl = _mm256_add_ps(_mm256_load_ps(&rhs[i * nperel]), rhv);
        __m256 mv = _mm256_mul_ps(_mm256_set1_ps(.5), _mm256_add_ps(lhl, rhl));
        __m256 lv0 = _mm256_mul_ps(lhl, Sleef_logf8_u35(_mm256_div_ps(lhl, mv)));
        __m256 rv0 = _mm256_mul_ps(rhl, Sleef_logf8_u35(_mm256_div_ps(rhl, mv)));
        sum = _mm256_add_ps(sum, _mm256_add_ps(lv0, rv0));
    }
    ret += .5 * (broadcast_reduce_add_si256_ps(sum)[0]);
    for(i *= nperel;i < n; ++i) {
        float xv = lhs[i] + lhinc;
        float yv = rhs[i] + rhinc;
        float mnv = (xv + yv) * .5;
        ret += .5 * (xv * log(xv / mnv) + yv * log(yv / mnv));
    }

#elif __SSE2__
    const size_t nperel = sizeof(__m128) / sizeof(float);
    __m128 sum = _mm_setzero_ps();
    size_t nsimd = n / nperel;
    __m128 lhv = _mm_set1_ps(lhinc);
    __m128 rhv = _mm_set1_ps(rhinc);
    #pragma GCC unroll 4
    for(i = 0; i < nsimd; ++i) {
        __m128 lhl = _mm_add_ps(_mm_load_ps(&lhs[i * nperel]), lhv);
        __m128 rhl = _mm_add_ps(_mm_load_ps(&rhs[i * nperel]), rhv);
        __m128 mv = _mm_mul_ps(_mm_set1_ps(.5), _mm_add_ps(lhl, rhl));
        __m128 lv0 = _mm_mul_ps(lhl, Sleef_logf4_u35(_mm_div_ps(lhl, mv)));
        __m128 rv0 = _mm_mul_ps(rhl, Sleef_logf4_u35(_mm_div_ps(rhl, mv)));
        sum = _mm_add_ps(sum, _mm_add_ps(lv0, rv0));
    }
    ret += .5 * (sum[0] + sum[1] + sum[2] + sum[3]);
    for(i *= nperel;i < n; ++i) {
        float xv = lhs[i] + lhinc;
        float yv = rhs[i] + rhinc;
        float mnv = (xv + yv) * .5;
        ret += .5 * (xv * log(xv / mnv) + yv * log(yv / mnv));
    }
#else
    float lhsum = 0.f, rhsum = 0.f;
    for(;i < n; ++i) {
        float lhv = lhs[i] + lhinc;
        float rhv = rhs[i] + rhinc;
        float miv = (lhv + rhv) * .5;
        lhsum += lhv * log(lhv / miv);
        rhsum += rhv * log(rhv / miv);
        ret += lhv * log(lhv / miv) + rhv * log(rhv / miv);
    }
    ret *= .5;
#endif
    return ret;
}
LIBKL_API double jsd_reduce_aligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, size_t n, double lhinc, double rhinc)
{
    double ret = 0.;
    size_t i = 0;
#if __AVX512F__
    const size_t nperel = sizeof(__m512d) / sizeof(double);
    __m512d lhsum = _mm512_setzero_pd(), rhsum = _mm512_setzero_pd();
    size_t nsimd = n / nperel;
    //
    __m512d lhv = _mm512_set1_pd(lhinc);
    __m512d rhv = _mm512_set1_pd(rhinc);
    #pragma GCC unroll 4
    for(i = 0; i < nsimd; ++i) {
        __m512d lhl = _mm512_add_pd(_mm512_load_pd(&lhs[i * nperel]), lhv);
        __m512d rhl = _mm512_add_pd(_mm512_load_pd(&rhs[i * nperel]), rhv);
        __m512d mv = _mm512_mul_pd(_mm512_set1_pd(.5), _mm512_add_pd(lhl, rhl));
        __m512d lv0 = _mm512_mul_pd(lhl, Sleef_logd8_u35(_mm512_div_pd(lhl, mv)));
        __m512d rv0 = _mm512_mul_pd(rhl, Sleef_logd8_u35(_mm512_div_pd(rhl, mv)));
        lhsum = _mm512_add_pd(lhsum, lv0);
        rhsum = _mm512_add_pd(rhsum, rv0);
    }
    ret += .5 * (_mm512_reduce_add_pd(lhsum) + _mm512_reduce_add_pd(rhsum));
    for(i *= nperel;i < n; ++i) {
        double xv = lhs[i] + lhinc;
        double yv = rhs[i] + rhinc;
        double mnv = (xv + yv) * .5;
        ret += .5 * (xv * log(xv / mnv) + yv * log(yv / mnv));
    }
#elif __AVX2__
    const size_t nperel = sizeof(__m256d) / sizeof(double);
    __m256d sum = _mm256_setzero_pd();
    size_t nsimd = n / nperel;
    __m256d lhv = _mm256_set1_pd(lhinc);
    __m256d rhv = _mm256_set1_pd(rhinc);
    #pragma GCC unroll 4
    for(i = 0; i < nsimd; ++i) {
        __m256d lhl = _mm256_add_pd(_mm256_load_pd(&lhs[i * nperel]), lhv);
        __m256d rhl = _mm256_add_pd(_mm256_load_pd(&rhs[i * nperel]), rhv);
        __m256d mv = _mm256_mul_pd(_mm256_set1_pd(.5), _mm256_add_pd(lhl, rhl));
        __m256d lv0 = _mm256_mul_pd(lhl, Sleef_logd4_u35(_mm256_div_pd(lhl, mv)));
        __m256d rv0 = _mm256_mul_pd(rhl, Sleef_logd4_u35(_mm256_div_pd(rhl, mv)));
        sum = _mm256_add_pd(sum, _mm256_add_pd(lv0, rv0));
    }
    ret += .5 * hsum_double_avx(sum);
    for(i *= nperel;i < n; ++i) {
        double xv = lhs[i] + lhinc;
        double yv = rhs[i] + rhinc;
        double mnv = (xv + yv) * .5;
        ret += .5 * (xv * log(xv / mnv) + yv * log(yv / mnv));
    }

#elif __SSE2__
    const size_t nperel = sizeof(__m128d) / sizeof(double);
    __m128d sum = _mm_setzero_pd();
    size_t nsimd = n / nperel;
    __m128d lhv = _mm_set1_pd(lhinc);
    __m128d rhv = _mm_set1_pd(rhinc);
    #pragma GCC unroll 4
    for(i = 0; i < nsimd; ++i) {
        __m128d lhl = _mm_add_pd(_mm_load_pd(&lhs[i * nperel]), lhv);
        __m128d rhl = _mm_add_pd(_mm_load_pd(&rhs[i * nperel]), rhv);
        __m128d mv = _mm_mul_pd(_mm_set1_pd(.5), _mm_add_pd(lhl, rhl));
        __m128d lv0 = _mm_mul_pd(lhl, Sleef_logd2_u35(_mm_div_pd(lhl, mv)));
        __m128d rv0 = _mm_mul_pd(rhl, Sleef_logd2_u35(_mm_div_pd(rhl, mv)));
        sum = _mm_add_pd(sum, _mm_add_pd(lv0, rv0));
    }
    ret += .5 * (sum[0] + sum[1]);
    for(i *= nperel;i < n; ++i) {
        double xv = lhs[i] + lhinc;
        double yv = rhs[i] + rhinc;
        double mnv = (xv + yv) * .5;
        ret += .5 * (xv * log(xv / mnv) + yv * log(yv / mnv));
    }
#else
    double lhsum = 0., rhsum = 0.;
    for(;i < n; ++i) {
        double lhv = lhs[i] + lhinc;
        double rhv = rhs[i] + rhinc;
        double miv = (lhv + rhv) * .5;
        lhsum += lhv * log(lhv / miv);
        rhsum += rhv * log(rhv / miv);
        ret += lhv * log(lhv / miv) + rhv * log(rhv / miv);
    }
    ret *= .5;
#endif
    return ret;
}
LIBKL_API double llr_reduce_aligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, size_t n, double lambda, double lhinc, double rhinc)
{
    const double m1l = 1. - lambda;
    double ret = 0.;
    size_t i = 0;
#if __AVX512F__
    const size_t nperel = sizeof(__m512d) / sizeof(double);
    __m512d lhsum = _mm512_setzero_pd(), rhsum = _mm512_setzero_pd();
    size_t nsimd = n / nperel;
    //
    __m512d vlambda = _mm512_set1_pd(lambda), vm1l = _mm512_set1_pd(m1l);
    __m512d lhv = _mm512_set1_pd(lhinc);
    __m512d rhv = _mm512_set1_pd(rhinc);
    #pragma GCC unroll 4
    for(i = 0; i < nsimd; ++i) {
        __m512d lhl = _mm512_add_pd(_mm512_load_pd(&lhs[i * nperel]), lhv);
        __m512d rhl = _mm512_add_pd(_mm512_load_pd(&rhs[i * nperel]), rhv);
        __m512d mv = _mm512_fmadd_pd(vlambda, lhl,  _mm512_mul_pd(vm1l, rhl));
        lhsum = _mm512_fmadd_pd(lhl, Sleef_logd8_u35(_mm512_div_pd(lhl, mv)), lhsum);
        rhsum = _mm512_fmadd_pd(rhl, Sleef_logd8_u35(_mm512_div_pd(rhl, mv)), rhsum);
    }
    ret += lambda * _mm512_reduce_add_pd(lhsum) + m1l * _mm512_reduce_add_pd(rhsum);
    for(i *= nperel;i < n; ++i) {
        double xv = lhs[i] + lhinc;
        double yv = rhs[i] + rhinc;
        double mnv = (xv * lambda + yv * m1l);
        ret += lambda * xv * log(xv / mnv) + m1l * yv * log(yv / mnv);
    }
#elif __AVX2__
    const size_t nperel = sizeof(__m256d) / sizeof(double);
    __m256d lhsum = _mm256_setzero_pd(), rhsum = _mm256_setzero_pd();
    size_t nsimd = n / nperel;
    //
    __m256d lhv = _mm256_set1_pd(lhinc), rhv = _mm256_set1_pd(rhinc);
    __m256d vlambda = _mm256_set1_pd(lambda), vm1l = _mm256_set1_pd(m1l);
    #pragma GCC unroll 4
    for(; i < nsimd; ++i) {
        __m256d lhsa = _mm256_add_pd(_mm256_load_pd(&lhs[i * nperel]), lhv);
        __m256d rhsa = _mm256_add_pd(_mm256_load_pd(&rhs[i * nperel]), rhv);
        __m256d mv = _mm256_add_pd(_mm256_mul_pd(vlambda, lhsa), _mm256_mul_pd(vm1l, rhsa));
        lhsum = _mm256_fmadd_pd(lhsa, Sleef_logd4_u35(_mm256_div_pd(lhsa, mv)), lhsum);
        rhsum = _mm256_fmadd_pd(rhsa, Sleef_logd4_u35(_mm256_div_pd(rhsa, mv)), rhsum);
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
        __m128d lhsa = _mm_add_pd(_mm_load_pd(&lhs[i * nperel]), _mm_set1_pd(lhinc));
        __m128d rhsa = _mm_add_pd(_mm_load_pd(&rhs[i * nperel]), _mm_set1_pd(rhinc));
#if LOG3
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
        ret += lambda * lhv * log(lhv / miv)
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


LIBKL_API double llr_reduce_unaligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, size_t n, double lambda, float lhinc, float rhinc)
{
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
#if LOG3
        __m512 lmv0 = Sleef_logf16_u35(_mm512_add_ps(_mm512_mul_ps(vlambda, lhl), _mm512_mul_ps(vm1l, rhl)));
        __m512 lv0 = _mm512_mul_ps(lhl, _mm512_sub_ps(Sleef_logf16_u35(lhl), lmv0));
        __m512 rv0 = _mm512_mul_ps(rhl, _mm512_sub_ps(Sleef_logf16_u35(rhl), rmv0));
#else
        __m512 mv = _mm512_add_ps(_mm512_mul_ps(vlambda, lhl), _mm512_mul_ps(vm1l, rhl));
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
        float mnv = (xv * lambda + yv * m1l);
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
        __m256 mv = _mm256_add_ps(_mm256_mul_ps(vlambda, lhsa), _mm256_mul_ps(vm1l, rhsa));
        lhsum = _mm256_fmadd_ps(lhsa, Sleef_logf8_u35(_mm256_div_ps(lhsa, mv)), lhsum);
        rhsum = _mm256_fmadd_ps(rhsa, Sleef_logf8_u35(_mm256_div_ps(rhsa, mv)), rhsum);
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
    //
    __m128 vlambda = _mm_set1_ps(lambda), vm1l = _mm_set1_ps(m1l);
    for(; i < nsimd; ++i) {
        __m128 lhsa = _mm_add_ps(_mm_loadu_ps(&lhs[i * nperel]), _mm_set1_ps(lhinc));
        __m128 rhsa = _mm_add_ps(_mm_loadu_ps(&rhs[i * nperel]), _mm_set1_ps(rhinc));
        __m128 mv = _mm_add_ps(_mm_mul_ps(vlambda, lhsa), _mm_mul_ps(vm1l, rhsa));
        lhsum = _mm_fmadd_ps(lhsa, Sleef_logf4_u35(_mm_div_ps(lhsa, mv)), lhsum);
        rhsum = _mm_fmadd_ps(rhsa, Sleef_logf4_u35(_mm_div_ps(rhsa, mv)), rhsum);
    }
    ret += lambda * broadcast_reduce_add_si128_ps(lhsum)[0] + m1l * broadcast_reduce_add_si128_ps(rhsum)[0];
    for(i *= nperel;i < n; ++i) {
        float lhv = lhs[i] + lhinc;
        float rhv = rhs[i] + rhinc;
        float miv = lhv * lambda + rhv * m1l;
        ret += lambda * lhv * logf(lhv / miv)
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

LIBKL_API double llr_reduce_unaligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, size_t n, double lambda, double lhinc, double rhinc)
{
    const double m1l = 1. - lambda;
    double ret = 0.;
    size_t i = 0;
#if __AVX512F__
    const size_t nperel = sizeof(__m512d) / sizeof(double);
    __m512d lhsum = _mm512_setzero_pd(), rhsum = _mm512_setzero_pd();
    size_t nsimd = n / nperel;
    //
    __m512d vlambda = _mm512_set1_pd(lambda), vm1l = _mm512_set1_pd(m1l);
    __m512d lhv = _mm512_set1_pd(lhinc);
    __m512d rhv = _mm512_set1_pd(rhinc);
    #pragma GCC unroll 4
    for(i = 0; i < nsimd; ++i) {
        __m512d lhl = _mm512_add_pd(_mm512_loadu_pd(&lhs[i * nperel]), lhv);
        __m512d rhl = _mm512_add_pd(_mm512_loadu_pd(&rhs[i * nperel]), rhv);
        __m512d mv = _mm512_add_pd(_mm512_mul_pd(vlambda, lhl), _mm512_mul_pd(vm1l, rhl));
        lhsum = _mm512_fmadd_pd(lhl, Sleef_logd8_u35(_mm512_div_pd(lhl, mv)), lhsum);
        rhsum = _mm512_fmadd_pd(rhl, Sleef_logd8_u35(_mm512_div_pd(rhl, mv)), rhsum);
    }
    ret += lambda * _mm512_reduce_add_pd(lhsum) + m1l * _mm512_reduce_add_pd(rhsum);
    for(i *= nperel;i < n; ++i) {
        double xv = lhs[i] + lhinc;
        double yv = rhs[i] + rhinc;
        double mnv = (xv * lambda + yv * m1l);
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
        __m256d mv = _mm256_add_pd(_mm256_mul_pd(vlambda, lhsa), _mm256_mul_pd(vm1l, rhsa));
        lhsum = _mm256_fmadd_pd(lhsa, Sleef_logd4_u35(_mm256_div_pd(lhsa, mv)), lhsum);
        rhsum = _mm256_fmadd_pd(rhsa, Sleef_logd4_u35(_mm256_div_pd(rhsa, mv)), rhsum);
    }
    ret += lambda * hsum_double_avx(lhsum) + m1l * hsum_double_avx(rhsum);
    for(i = nsimd * nperel;i < n; ++i) {
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
        __m128d lhsa = _mm_add_pd(_mm_loadu_pd(&lhs[i * nperel]), _mm_set1_pd(lhinc));
        __m128d rhsa = _mm_add_pd(_mm_loadu_pd(&rhs[i * nperel]), _mm_set1_pd(rhinc));
#if LOG3
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
        ret += lambda * lhv * log(lhv / miv)
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

LIBKL_API double jsd_reduce_unaligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, size_t n, float lhinc, float rhinc)
{
    double ret = 0.;
    size_t i = 0;
#if __AVX512F__
    const size_t nperel = sizeof(__m512) / sizeof(float);
    __m512 sum = _mm512_setzero_ps();
    size_t nsimd = n / nperel;
    //
    __m512 lhv = _mm512_set1_ps(lhinc);
    __m512 rhv = _mm512_set1_ps(rhinc);
    #pragma GCC unroll 4
    for(i = 0; i < nsimd; ++i) {
        __m512 lhl = _mm512_add_ps(_mm512_loadu_ps(&lhs[i * nperel]), lhv);
        __m512 rhl = _mm512_add_ps(_mm512_loadu_ps(&rhs[i * nperel]), rhv);
        __m512 mv = _mm512_mul_ps(_mm512_set1_ps(.5), _mm512_add_ps(lhl, rhl));
        sum = _mm512_fmadd_ps(lhl, Sleef_logf16_u35(_mm512_div_ps(lhl, mv)), sum);
        sum = _mm512_fmadd_ps(rhl, Sleef_logf16_u35(_mm512_div_ps(rhl, mv)), sum);
    }
    ret += .5f * (_mm512_reduce_add_ps(sum));
    for(i *= nperel;i < n; ++i) {
        float xv = lhs[i] + lhinc;
        float yv = rhs[i] + rhinc;
        float mnv = (xv + yv) * .5f;
        ret += .5f * (xv * log(xv / mnv) + yv * log(yv / mnv));
    }
    i *= nperel;
#elif __AVX2__
    const size_t nperel = sizeof(__m256) / sizeof(float);
    __m256 sum = _mm256_setzero_ps();
    size_t nsimd = n / nperel;
    __m256 lhv = _mm256_set1_ps(lhinc);
    __m256 rhv = _mm256_set1_ps(rhinc);
    #pragma GCC unroll 4
    for(i = 0; i < nsimd; ++i) {
        __m256 lhl = _mm256_add_ps(_mm256_loadu_ps(&lhs[i * nperel]), lhv);
        __m256 rhl = _mm256_add_ps(_mm256_loadu_ps(&rhs[i * nperel]), rhv);
        __m256 mv = _mm256_mul_ps(_mm256_set1_ps(.5), _mm256_add_ps(lhl, rhl));
        sum = _mm256_fmadd_ps(lhl, Sleef_logf8_u35(_mm256_div_ps(lhl, mv)), sum);
        sum = _mm256_fmadd_ps(rhl, Sleef_logf8_u35(_mm256_div_ps(rhl, mv)), sum);
    }
    ret += .5 * (broadcast_reduce_add_si256_ps(sum)[0]);
    for(i *= nperel;i < n; ++i) {
        float xv = lhs[i] + lhinc;
        float yv = rhs[i] + rhinc;
        float mnv = (xv + yv) * .5;
        ret += .5 * (xv * log(xv / mnv) + yv * log(yv / mnv));
    }
    i *= nperel;

#elif __SSE2__
    const size_t nperel = sizeof(__m128) / sizeof(float);
    __m128 sum = _mm_setzero_ps();
    size_t nsimd = n / nperel;
    __m128 lhv = _mm_set1_ps(lhinc);
    __m128 rhv = _mm_set1_ps(rhinc);
    #pragma GCC unroll 4
    for(i = 0; i < nsimd; ++i) {
        __m128 lhl = _mm_add_ps(_mm_loadu_ps(&lhs[i * nperel]), lhv);
        __m128 rhl = _mm_add_ps(_mm_loadu_ps(&rhs[i * nperel]), rhv);
        __m128 mv = _mm_mul_ps(_mm_set1_ps(.5), _mm_add_ps(lhl, rhl));
        sum = _mm_fmadd_ps(lhl, Sleef_logf4_u35(_mm_div_ps(lhl, mv)), sum);
        sum = _mm_fmadd_ps(rhl, Sleef_logf4_u35(_mm_div_ps(rhl, mv)), sum);
    }
    ret += .5 * (sum[0] + sum[1] + sum[2] + sum[3]);
    i *= nperel;
#endif
    for(;i < n; ++i) {
        float lhv = lhs[i] + lhinc;
        float rhv = rhs[i] + rhinc;
        float miv = (lhv + rhv) * .5;
        ret += .5 * (lhv * log(lhv / miv) + rhv * log(rhv / miv));
    }
    return ret;
}

LIBKL_API double jsd_reduce_unaligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, size_t n, double lhinc, double rhinc)
{
    double ret = 0.;
    size_t i = 0;
#if __AVX512F__
    const size_t nperel = sizeof(__m512d) / sizeof(double);
    __m512d lhsum = _mm512_setzero_pd(), rhsum = _mm512_setzero_pd();
    size_t nsimd = n / nperel;
    //
    __m512d lhv = _mm512_set1_pd(lhinc);
    __m512d rhv = _mm512_set1_pd(rhinc);
    #pragma GCC unroll 4
    for(i = 0; i < nsimd; ++i) {
        __m512d lhl = _mm512_add_pd(_mm512_loadu_pd(&lhs[i * nperel]), lhv);
        __m512d rhl = _mm512_add_pd(_mm512_loadu_pd(&rhs[i * nperel]), rhv);
        __m512d mv = _mm512_mul_pd(_mm512_set1_pd(.5), _mm512_add_pd(lhl, rhl));
        lhsum = _mm512_fmadd_pd(lhl, Sleef_logd8_u35(_mm512_div_pd(lhl, mv)), lhsum);
        rhsum = _mm512_fmadd_pd(rhl, Sleef_logd8_u35(_mm512_div_pd(rhl, mv)), rhsum);
    }
    ret += .5 * (_mm512_reduce_add_pd(lhsum) + _mm512_reduce_add_pd(rhsum));
    for(i *= nperel;i < n; ++i) {
        double xv = lhs[i] + lhinc;
        double yv = rhs[i] + rhinc;
        double mnv = (xv + yv) * .5;
        ret += .5 * (xv * log(xv / mnv) + yv * log(yv / mnv));
    }
#elif __AVX2__
    const size_t nperel = sizeof(__m256d) / sizeof(double);
    __m256d sum = _mm256_setzero_pd();
    size_t nsimd = n / nperel;
    __m256d lhv = _mm256_set1_pd(lhinc);
    __m256d rhv = _mm256_set1_pd(rhinc);
    #pragma GCC unroll 4
    for(i = 0; i < nsimd; ++i) {
        __m256d lhl = _mm256_add_pd(_mm256_loadu_pd(&lhs[i * nperel]), lhv);
        __m256d rhl = _mm256_add_pd(_mm256_loadu_pd(&rhs[i * nperel]), rhv);
        __m256d mv = _mm256_mul_pd(_mm256_set1_pd(.5), _mm256_add_pd(lhl, rhl));
        sum = _mm256_fmadd_pd(lhl, Sleef_logd4_u35(_mm256_div_pd(lhl, mv)), sum);
        sum = _mm256_fmadd_pd(rhl, Sleef_logd4_u35(_mm256_div_pd(rhl, mv)), sum);
    }
    ret += .5 * hsum_double_avx(sum);
    for(i *= nperel;i < n; ++i) {
        double xv = lhs[i] + lhinc;
        double yv = rhs[i] + rhinc;
        double mnv = (xv + yv) * .5;
        ret += .5 * (xv * log(xv / mnv) + yv * log(yv / mnv));
    }

#elif __SSE2__
    const size_t nperel = sizeof(__m128d) / sizeof(double);
    __m128d sum = _mm_setzero_pd();
    size_t nsimd = n / nperel;
    __m128d lhv = _mm_set1_pd(lhinc);
    __m128d rhv = _mm_set1_pd(rhinc);
    #pragma GCC unroll 4
    for(i = 0; i < nsimd; ++i) {
        __m128d lhl = _mm_add_pd(_mm_loadu_pd(&lhs[i * nperel]), lhv);
        __m128d rhl = _mm_add_pd(_mm_loadu_pd(&rhs[i * nperel]), rhv);
        __m128d mv = _mm_mul_pd(_mm_set1_pd(.5), _mm_add_pd(lhl, rhl));
        sum = _mm_fmadd_pd(lhl, Sleef_logd2_u35(_mm_div_pd(lhl, mv)), sum);
        sum = _mm_fmadd_pd(rhl, Sleef_logd2_u35(_mm_div_pd(rhl, mv)), sum);
    }
    ret += .5 * (sum[0] + sum[1]);
    for(i *= nperel;i < n; ++i) {
        double xv = lhs[i] + lhinc;
        double yv = rhs[i] + rhinc;
        double mnv = (xv + yv) * .5;
        ret += .5 * (xv * log(xv / mnv) + yv * log(yv / mnv));
    }
#else
    for(;i < n; ++i) {
        double lhv = lhs[i] + lhinc;
        double rhv = rhs[i] + rhinc;
        double miv = (lhv + rhv) * .5;
        ret += lhv * log(lhv / miv) + rhv * log(rhv / miv);
    }
    ret *= .5;
#endif
    return ret;
}

#define __COSSIM_FUNC(T, VecT, LoadFnc, AddFnc, Set1Fnc, SetZeroFnc, Name, ReduceFnc, MulFnc, FMAFnc) \
LIBKL_API double Name(const T *const __restrict__ lhs, const T *const __restrict__ rhs, const size_t n, T lhmul, T rhmul, T lhi, T rhi) {\
    double ssumd = 0., lsumd = 0., rsumd = 0.;\
    const size_t nper = sizeof(VecT) / sizeof(T);\
    const size_t nsimd = n / nper;\
    const size_t nsimd4 = (nsimd >> 2) << 2;\
    VecT lsum = SetZeroFnc(), rsum = SetZeroFnc(), ssum = SetZeroFnc();\
    size_t i;\
    for(i = 0; i < nsimd4; i += 4) {\
        const VecT lhv0 = FMAFnc(LoadFnc(lhs + (i * nper)), Set1Fnc(lhmul), Set1Fnc(lhi));\
        const VecT lhv1 = FMAFnc(LoadFnc(lhs + ((i + 1) * nper)), Set1Fnc(lhmul), Set1Fnc(lhi));\
        const VecT lhv2 = FMAFnc(LoadFnc(lhs + ((i + 2) * nper)), Set1Fnc(lhmul), Set1Fnc(lhi));\
        const VecT lhv3 = FMAFnc(LoadFnc(lhs + ((i + 3) * nper)), Set1Fnc(lhmul), Set1Fnc(lhi));\
        const VecT rhv0 = FMAFnc(LoadFnc(rhs + (i * nper)), Set1Fnc(rhmul), Set1Fnc(rhi));\
        const VecT rhv1 = FMAFnc(LoadFnc(rhs + ((i + 1) * nper)), Set1Fnc(rhmul), Set1Fnc(rhi));\
        const VecT rhv2 = FMAFnc(LoadFnc(rhs + ((i + 2) * nper)), Set1Fnc(rhmul), Set1Fnc(rhi));\
        const VecT rhv3 = FMAFnc(LoadFnc(rhs + ((i + 3) * nper)), Set1Fnc(rhmul), Set1Fnc(rhi));\
        lsum = FMAFnc(lhv0, lhv0, lsum);\
        lsum = FMAFnc(lhv1, lhv1, lsum);\
        lsum = FMAFnc(lhv2, lhv2, lsum);\
        lsum = FMAFnc(lhv3, lhv3, lsum);\
        rsum = FMAFnc(rhv0, rhv0, rsum);\
        rsum = FMAFnc(rhv1, rhv1, rsum);\
        rsum = FMAFnc(rhv2, rhv2, rsum);\
        rsum = FMAFnc(rhv3, rhv3, rsum);\
        ssum = FMAFnc(lhv0, rhv0, ssum);\
        ssum = FMAFnc(lhv1, rhv1, ssum);\
        ssum = FMAFnc(lhv2, rhv2, ssum);\
        ssum = FMAFnc(lhv3, rhv3, ssum);\
    }\
    for(; i < nsimd; ++i) {\
        const VecT lhv = FMAFnc(LoadFnc(lhs + (i * nper)), Set1Fnc(lhmul), Set1Fnc(lhi));\
        const VecT rhv = FMAFnc(LoadFnc(rhs + (i * nper)), Set1Fnc(rhmul), Set1Fnc(rhi));\
        lsum = FMAFnc(lhv, lhv, lsum); rsum = FMAFnc(rhv, rhv, rsum); ssum = FMAFnc(lhv, rhv, ssum);\
    }\
    ssumd = ReduceFnc(ssum);\
    lsumd = ReduceFnc(lsum);\
    rsumd = ReduceFnc(rsum);\
    for(i = nsimd * nper;i < n; ++i) {\
        T lh = fabs(lhs[i] * lhmul + lhi);\
        T rh = fabs(rhs[i] * rhmul + rhi);\
        ssumd += lh * rh; lsumd += lh * lh; rsumd += rh * rh;\
    }\
    return ssumd / sqrt(lsumd * rsumd);\
}

#define __SQRL2_FUNC_NONORM(T, VecT, LoadFnc, AddFnc, Set1Fnc, SqrtFnc, SetZeroFnc, Name, ReduceFnc, MulFnc, SubFnc, FMAFnc) \
LIBKL_API double Name(const T *const __restrict__ lhs, const T *const __restrict__ rhs, const size_t n) {\
    double ret = 0.;\
    const size_t nper = sizeof(VecT) / sizeof(T);\
    const size_t nsimd = n / nper;\
    VecT sum = SetZeroFnc();\
    size_t i;\
    _Pragma("GCC unroll 8")\
    for(i = 0; i < nsimd; ++i) {\
        VecT lhv = LoadFnc(lhs + (i * nper));\
        VecT rhv = LoadFnc(rhs + (i * nper));\
        VecT diffv = SubFnc(lhv, rhv);\
        sum = FMAFnc(diffv, diffv, sum);\
    }\
    ret = ReduceFnc(sum);\
    for(i *= nper;i < n; ++i) {\
        double v = (lhs[i] - rhs[i]);\
        ret += v * v;\
    }\
    return ret;\
}

#define __SQRL2_FUNC(T, VecT, LoadFnc, AddFnc, Set1Fnc, SqrtFnc, SetZeroFnc, Name, ReduceFnc, MulFnc, SubFnc, FMAFnc) \
LIBKL_API double Name(const T *const __restrict__ lhs, const T *const __restrict__ rhs, const size_t n, T lhmul, T rhmul, T lhi, T rhi) {\
    double ret = 0.;\
    const size_t nper = sizeof(VecT) / sizeof(T);\
    const size_t nsimd = n / nper;\
    VecT sum = SetZeroFnc();\
    size_t i;\
    _Pragma("GCC unroll 4")\
    for(i = 0; i < nsimd; ++i) {\
        VecT lhv = FMAFnc(LoadFnc(lhs + (i * nper)), Set1Fnc(lhmul), Set1Fnc(lhi));\
        VecT rhv = FMAFnc(LoadFnc(rhs + (i * nper)), Set1Fnc(rhmul), Set1Fnc(rhi));\
        VecT diffv = SubFnc(lhv, rhv);\
        sum = FMAFnc(diffv, diffv, sum);\
    }\
    ret = ReduceFnc(sum);\
    for(i *= nper;i < n; ++i) {\
        double v = (lhs[i] * lhmul + lhi) - (rhs[i] * rhmul + rhi);\
        ret += v * v;\
    }\
    return ret;\
}

#define __TVD_FUNC(T, VecT, LoadFnc, AddFnc, Set1Fnc, SqrtFnc, SetZeroFnc, Name, ReduceFnc, MulFnc, SubFnc, AbsFnc, FMAFnc) \
LIBKL_API double Name(const T *const __restrict__ lhs, const T *const __restrict__ rhs, const size_t n, T lhmul, T rhmul, T lhi, T rhi) {\
    double ret = 0.;\
    const size_t nper = sizeof(VecT) / sizeof(T);\
    const size_t nsimd = n / nper;\
    VecT sum = SetZeroFnc();\
    size_t i;\
    _Pragma("GCC unroll 4")\
    for(i = 0; i < nsimd; ++i) {\
        VecT lhv = FMAFnc(LoadFnc(lhs + (i * nper)), Set1Fnc(lhmul), Set1Fnc(lhi));\
        VecT rhv = FMAFnc(LoadFnc(rhs + (i * nper)), Set1Fnc(rhmul), Set1Fnc(rhi));\
        VecT diffv = SubFnc(lhv, rhv);\
        sum = AddFnc(AbsFnc(diffv), sum);\
    }\
    ret = ReduceFnc(sum);\
    for(i *= nper;i < n; ++i) {\
        ret += fabs((lhs[i] * lhmul + lhi) - (rhs[i] * rhmul + rhi));\
    }\
    return ret * .5;\
}

#define __HELLDIST_FUNC(T, VecT, LoadFnc, AddFnc, Set1Fnc, SqrtFnc, SetZeroFnc, Name, ReduceFnc, MulFnc, SubFnc, FMAFnc) \
LIBKL_API double Name(const T *const __restrict__ lhs, const T *const __restrict__ rhs, const size_t n, T lhmul, T rhmul, T lhi, T rhi) {\
    double ret = 0.;\
    const size_t nper = sizeof(VecT) / sizeof(T);\
    const size_t nsimd = n / nper;\
    VecT sum = SetZeroFnc();\
    size_t i;\
    _Pragma("GCC unroll 4")\
    for(i = 0; i < nsimd; ++i) {\
        VecT lhv = SqrtFnc(FMAFnc(LoadFnc(lhs + (i * nper)), Set1Fnc(lhmul), Set1Fnc(lhi)));\
        VecT rhv = SqrtFnc(FMAFnc(LoadFnc(rhs + (i * nper)), Set1Fnc(rhmul), Set1Fnc(rhi)));\
        VecT diffv = SubFnc(lhv, rhv);\
        sum = FMAFnc(diffv, diffv, sum);\
    }\
    ret = ReduceFnc(sum);\
    for(i *= nper;i < n; ++i) {\
        double v = sqrt(lhs[i] * lhmul + lhi) - sqrt(rhs[i] * rhmul + rhi);\
        ret += v * v;\
    }\
    return ret;\
}

#define __D_BSIM_FUNC(T, VecT, LoadFnc, AddFnc, Set1Fnc, SqrtFnc, SetZeroFnc, Name, ReduceFnc, MulFnc, FmaFnc) \
LIBKL_API double Name(const T *const __restrict__ lhs, const T *const __restrict__ rhs, const size_t n, T lhmul, T rhmul, T lhi, T rhi) {\
    double ret = 0.;\
    const size_t nper = sizeof(VecT) / sizeof(T);\
    const size_t nsimd = n / nper;\
    VecT sum = SetZeroFnc();\
    size_t i;\
    _Pragma("GCC unroll 4")\
    for(i = 0; i < nsimd; ++i) {\
        VecT lhv = FmaFnc(LoadFnc(lhs + (i * nper)), Set1Fnc(lhmul), Set1Fnc(lhi));\
        VecT rhv = FmaFnc(LoadFnc(rhs + (i * nper)), Set1Fnc(rhmul), Set1Fnc(rhi));\
        VecT mulv = MulFnc(lhv, rhv);\
        VecT sqrv = SqrtFnc(mulv);\
        sum = AddFnc(sum, sqrv);\
    }\
    ret = ReduceFnc(sum);\
    for(i *= nper;i < n; ++i) {\
        ret += sqrt((lhs[i] * lhmul + lhi) * (rhs[i] * rhmul + rhi));\
    }\
    return ret;\
}


#ifdef __AVX512F__
__D_BSIM_FUNC(double, __m512d, _mm512_loadu_pd, _mm512_add_pd, _mm512_set1_pd, Sleef_sqrtd8_u35, _mm512_setzero_pd, dbhattd_reduce_unaligned_avx512, _mm512_reduce_add_pdd, _mm512_mul_pd, _mm512_fmadd_pd)
__D_BSIM_FUNC(double, __m512d, _mm512_load_pd, _mm512_add_pd, _mm512_set1_pd, Sleef_sqrtd8_u35, _mm512_setzero_pd, dbhattd_reduce_aligned_avx512, _mm512_reduce_add_pdd, _mm512_mul_pd, _mm512_fmadd_pd)
__D_BSIM_FUNC(float, __m512, _mm512_loadu_ps, _mm512_add_ps, _mm512_set1_ps, Sleef_sqrtf16_u35, _mm512_setzero_ps, fbhattd_reduce_unaligned_avx512, _mm512_reduce_add_psf, _mm512_mul_ps, _mm512_fmadd_ps)
__D_BSIM_FUNC(float, __m512, _mm512_load_ps, _mm512_add_ps, _mm512_set1_ps, Sleef_sqrtf16_u35, _mm512_setzero_ps, fbhattd_reduce_aligned_avx512, _mm512_reduce_add_psf, _mm512_mul_ps, _mm512_fmadd_ps)
__HELLDIST_FUNC(double, __m512d, _mm512_loadu_pd, _mm512_add_pd, _mm512_set1_pd, Sleef_sqrtd8_u35, _mm512_setzero_pd, dhelld_reduce_unaligned_avx512, _mm512_reduce_add_pdd, _mm512_mul_pd, _mm512_sub_pd, _mm512_fmadd_pd)
__HELLDIST_FUNC(double, __m512d, _mm512_load_pd, _mm512_add_pd, _mm512_set1_pd, Sleef_sqrtd8_u35, _mm512_setzero_pd, dhelld_reduce_aligned_avx512, _mm512_reduce_add_pdd, _mm512_mul_pd, _mm512_sub_pd, _mm512_fmadd_pd)
__HELLDIST_FUNC(float, __m512, _mm512_loadu_ps, _mm512_add_ps, _mm512_set1_ps, Sleef_sqrtf16_u35, _mm512_setzero_ps, fhelld_reduce_unaligned_avx512, _mm512_reduce_add_psf, _mm512_mul_ps, _mm512_sub_ps, _mm512_fmadd_ps)
__HELLDIST_FUNC(float, __m512, _mm512_load_ps, _mm512_add_ps, _mm512_set1_ps, Sleef_sqrtf16_u35, _mm512_setzero_ps, fhelld_reduce_aligned_avx512, _mm512_reduce_add_psf, _mm512_mul_ps, _mm512_sub_ps, _mm512_fmadd_ps)
__SQRL2_FUNC(double, __m512d, _mm512_loadu_pd, _mm512_add_pd, _mm512_set1_pd, Sleef_sqrtd8_u35, _mm512_setzero_pd, dsqrl2_reduce_unaligned_avx512, _mm512_reduce_add_pdd, _mm512_mul_pd, _mm512_sub_pd, _mm512_fmadd_pd)
__SQRL2_FUNC(double, __m512d, _mm512_load_pd, _mm512_add_pd, _mm512_set1_pd, Sleef_sqrtd8_u35, _mm512_setzero_pd, dsqrl2_reduce_aligned_avx512, _mm512_reduce_add_pdd, _mm512_mul_pd, _mm512_sub_pd, _mm512_fmadd_pd)
__SQRL2_FUNC(float, __m512, _mm512_loadu_ps, _mm512_add_ps, _mm512_set1_ps, Sleef_sqrtf16_u35, _mm512_setzero_ps, fsqrl2_reduce_unaligned_avx512, _mm512_reduce_add_psf, _mm512_mul_ps, _mm512_sub_ps, _mm512_fmadd_ps)
__SQRL2_FUNC(float, __m512, _mm512_load_ps, _mm512_add_ps, _mm512_set1_ps, Sleef_sqrtf16_u35, _mm512_setzero_ps, fsqrl2_reduce_aligned_avx512, _mm512_reduce_add_psf, _mm512_mul_ps, _mm512_sub_ps, _mm512_fmadd_ps)
__SQRL2_FUNC_NONORM(double, __m512d, _mm512_loadu_pd, _mm512_add_pd, _mm512_set1_pd, Sleef_sqrtd8_u35, _mm512_setzero_pd, dnnsqrl2_reduce_unaligned_avx512, _mm512_reduce_add_pdd, _mm512_mul_pd, _mm512_sub_pd, _mm512_fmadd_pd)
__SQRL2_FUNC_NONORM(double, __m512d, _mm512_load_pd, _mm512_add_pd, _mm512_set1_pd, Sleef_sqrtd8_u35, _mm512_setzero_pd, dnnsqrl2_reduce_aligned_avx512, _mm512_reduce_add_pdd, _mm512_mul_pd, _mm512_sub_pd, _mm512_fmadd_pd)
__SQRL2_FUNC_NONORM(float, __m512, _mm512_loadu_ps, _mm512_add_ps, _mm512_set1_ps, Sleef_sqrtf16_u35, _mm512_setzero_ps, fnnsqrl2_reduce_unaligned_avx512, _mm512_reduce_add_psf, _mm512_mul_ps, _mm512_sub_ps, _mm512_fmadd_ps)
__SQRL2_FUNC_NONORM(float, __m512, _mm512_load_ps, _mm512_add_ps, _mm512_set1_ps, Sleef_sqrtf16_u35, _mm512_setzero_ps, fnnsqrl2_reduce_aligned_avx512, _mm512_reduce_add_psf, _mm512_mul_ps, _mm512_sub_ps, _mm512_fmadd_ps)
__TVD_FUNC(double, __m512d, _mm512_loadu_pd, _mm512_add_pd, _mm512_set1_pd, Sleef_sqrtd8_u35, _mm512_setzero_pd, dtvd_reduce_unaligned_avx512, _mm512_reduce_add_pdd, _mm512_mul_pd, _mm512_sub_pd, _mm512_abs_pd, _mm512_fmadd_pd)
__TVD_FUNC(double, __m512d, _mm512_load_pd, _mm512_add_pd, _mm512_set1_pd, Sleef_sqrtd8_u35, _mm512_setzero_pd, dtvd_reduce_aligned_avx512, _mm512_reduce_add_pdd, _mm512_mul_pd, _mm512_sub_pd, _mm512_abs_pd, _mm512_fmadd_pd)
__TVD_FUNC(float, __m512, _mm512_loadu_ps, _mm512_add_ps, _mm512_set1_ps, Sleef_sqrtf16_u35, _mm512_setzero_ps, ftvd_reduce_unaligned_avx512, _mm512_reduce_add_psf, _mm512_mul_ps, _mm512_sub_ps, _mm512_abs_ps, _mm512_fmadd_ps)
__TVD_FUNC(float, __m512, _mm512_load_ps, _mm512_add_ps, _mm512_set1_ps, Sleef_sqrtf16_u35, _mm512_setzero_ps, ftvd_reduce_aligned_avx512, _mm512_reduce_add_psf, _mm512_mul_ps, _mm512_sub_ps, _mm512_abs_ps, _mm512_fmadd_ps)
__COSSIM_FUNC(double, __m512d, _mm512_loadu_pd, _mm512_add_pd, _mm512_set1_pd, _mm512_setzero_pd, dcossim_reduce_unaligned_avx512, _mm512_reduce_add_pdd, _mm512_mul_pd, _mm512_fmadd_pd)
__COSSIM_FUNC(double, __m512d, _mm512_load_pd, _mm512_add_pd, _mm512_set1_pd, _mm512_setzero_pd, dcossim_reduce_aligned_avx512, _mm512_reduce_add_pdd, _mm512_mul_pd, _mm512_fmadd_pd)
__COSSIM_FUNC(float, __m512, _mm512_loadu_ps, _mm512_add_ps, _mm512_set1_ps, _mm512_setzero_ps, fcossim_reduce_unaligned_avx512, _mm512_reduce_add_psf, _mm512_mul_ps, _mm512_fmadd_ps)
__COSSIM_FUNC(float, __m512, _mm512_load_ps, _mm512_add_ps, _mm512_set1_ps, _mm512_setzero_ps, fcossim_reduce_aligned_avx512, _mm512_reduce_add_psf, _mm512_mul_ps, _mm512_fmadd_ps)
#endif
#ifdef __AVX2__
__D_BSIM_FUNC(double, __m256d, _mm256_loadu_pd, _mm256_add_pd, _mm256_set1_pd, Sleef_sqrtd4_u35, _mm256_setzero_pd, dbhattd_reduce_unaligned_avx256, hsum_double_avx, _mm256_mul_pd,_mm256_fmadd_pd)
__D_BSIM_FUNC(double, __m256d, _mm256_load_pd, _mm256_add_pd, _mm256_set1_pd, Sleef_sqrtd4_u35, _mm256_setzero_pd, dbhattd_reduce_aligned_avx256, hsum_double_avx, _mm256_mul_pd,_mm256_fmadd_pd)
__D_BSIM_FUNC(float, __m256, _mm256_loadu_ps, _mm256_add_ps, _mm256_set1_ps, Sleef_sqrtf8_u35, _mm256_setzero_ps, fbhattd_reduce_unaligned_avx256, broadcast_reduce_add_si256_psf, _mm256_mul_ps,_mm256_fmadd_ps)
__D_BSIM_FUNC(float, __m256, _mm256_load_ps, _mm256_add_ps, _mm256_set1_ps, Sleef_sqrtf8_u35, _mm256_setzero_ps, fbhattd_reduce_aligned_avx256, broadcast_reduce_add_si256_psf, _mm256_mul_ps,_mm256_fmadd_ps)
__HELLDIST_FUNC(double, __m256d, _mm256_loadu_pd, _mm256_add_pd, _mm256_set1_pd, Sleef_sqrtd4_u35, _mm256_setzero_pd, dhelld_reduce_unaligned_avx256, hsum_double_avx, _mm256_mul_pd, _mm256_sub_pd, _mm256_fmadd_pd)
__HELLDIST_FUNC(double, __m256d, _mm256_load_pd, _mm256_add_pd, _mm256_set1_pd, Sleef_sqrtd4_u35, _mm256_setzero_pd, dhelld_reduce_aligned_avx256, hsum_double_avx, _mm256_mul_pd, _mm256_sub_pd, _mm256_fmadd_pd)
__HELLDIST_FUNC(float, __m256, _mm256_loadu_ps, _mm256_add_ps, _mm256_set1_ps, Sleef_sqrtf8_u35, _mm256_setzero_ps, fhelld_reduce_unaligned_avx256, broadcast_reduce_add_si256_psf, _mm256_mul_ps, _mm256_sub_ps, _mm256_fmadd_ps)
__HELLDIST_FUNC(float, __m256, _mm256_load_ps, _mm256_add_ps, _mm256_set1_ps, Sleef_sqrtf8_u35, _mm256_setzero_ps, fhelld_reduce_aligned_avx256, broadcast_reduce_add_si256_psf, _mm256_mul_ps, _mm256_sub_ps, _mm256_fmadd_ps)
__SQRL2_FUNC(double, __m256d, _mm256_loadu_pd, _mm256_add_pd, _mm256_set1_pd, Sleef_sqrtd4_u35, _mm256_setzero_pd, dsqrl2_reduce_unaligned_avx256, hsum_double_avx, _mm256_mul_pd, _mm256_sub_pd, _mm256_fmadd_pd)
__SQRL2_FUNC(double, __m256d, _mm256_load_pd, _mm256_add_pd, _mm256_set1_pd, Sleef_sqrtd4_u35, _mm256_setzero_pd, dsqrl2_reduce_aligned_avx256, hsum_double_avx, _mm256_mul_pd, _mm256_sub_pd, _mm256_fmadd_pd)
__SQRL2_FUNC(float, __m256, _mm256_loadu_ps, _mm256_add_ps, _mm256_set1_ps, Sleef_sqrtf8_u35, _mm256_setzero_ps, fsqrl2_reduce_unaligned_avx256, broadcast_reduce_add_si256_psf, _mm256_mul_ps, _mm256_sub_ps, _mm256_fmadd_ps)
__SQRL2_FUNC(float, __m256, _mm256_load_ps, _mm256_add_ps, _mm256_set1_ps, Sleef_sqrtf8_u35, _mm256_setzero_ps, fsqrl2_reduce_aligned_avx256, broadcast_reduce_add_si256_psf, _mm256_mul_ps, _mm256_sub_ps, _mm256_fmadd_ps)
__SQRL2_FUNC_NONORM(double, __m256d, _mm256_loadu_pd, _mm256_add_pd, _mm256_set1_pd, Sleef_sqrtd4_u35, _mm256_setzero_pd, dnnsqrl2_reduce_unaligned_avx256, hsum_double_avx, _mm256_mul_pd, _mm256_sub_pd, _mm256_fmadd_pd)
__SQRL2_FUNC_NONORM(double, __m256d, _mm256_load_pd, _mm256_add_pd, _mm256_set1_pd, Sleef_sqrtd4_u35, _mm256_setzero_pd, dnnsqrl2_reduce_aligned_avx256, hsum_double_avx, _mm256_mul_pd, _mm256_sub_pd, _mm256_fmadd_pd)
__SQRL2_FUNC_NONORM(float, __m256, _mm256_loadu_ps, _mm256_add_ps, _mm256_set1_ps, Sleef_sqrtf8_u35, _mm256_setzero_ps, fnnsqrl2_reduce_unaligned_avx256, broadcast_reduce_add_si256_psf, _mm256_mul_ps, _mm256_sub_ps, _mm256_fmadd_ps)
__SQRL2_FUNC_NONORM(float, __m256, _mm256_load_ps, _mm256_add_ps, _mm256_set1_ps, Sleef_sqrtf8_u35, _mm256_setzero_ps, fnnsqrl2_reduce_aligned_avx256, broadcast_reduce_add_si256_psf, _mm256_mul_ps, _mm256_sub_ps, _mm256_fmadd_ps)
__TVD_FUNC(double, __m256d, _mm256_loadu_pd, _mm256_add_pd, _mm256_set1_pd, Sleef_sqrtd4_u35, _mm256_setzero_pd, dtvd_reduce_unaligned_avx256, hsum_double_avx, _mm256_mul_pd, _mm256_sub_pd, _mm256_kl_abs_pd, _mm256_fmadd_pd)
__TVD_FUNC(double, __m256d, _mm256_load_pd, _mm256_add_pd, _mm256_set1_pd, Sleef_sqrtd4_u35, _mm256_setzero_pd, dtvd_reduce_aligned_avx256, hsum_double_avx, _mm256_mul_pd, _mm256_sub_pd, _mm256_kl_abs_pd, _mm256_fmadd_pd)
__TVD_FUNC(float, __m256, _mm256_loadu_ps, _mm256_add_ps, _mm256_set1_ps, Sleef_sqrtf8_u35, _mm256_setzero_ps, ftvd_reduce_unaligned_avx256, broadcast_reduce_add_si256_psf, _mm256_mul_ps, _mm256_sub_ps, _mm256_kl_abs_ps, _mm256_fmadd_ps)
__TVD_FUNC(float, __m256, _mm256_load_ps, _mm256_add_ps, _mm256_set1_ps, Sleef_sqrtf8_u35, _mm256_setzero_ps, ftvd_reduce_aligned_avx256, broadcast_reduce_add_si256_psf, _mm256_mul_ps, _mm256_sub_ps, _mm256_kl_abs_ps, _mm256_fmadd_ps)
__COSSIM_FUNC(double, __m256d, _mm256_loadu_pd, _mm256_add_pd, _mm256_set1_pd, _mm256_setzero_pd, dcossim_reduce_unaligned_avx256, hsum_double_avx, _mm256_mul_pd,_mm256_fmadd_pd)
__COSSIM_FUNC(double, __m256d, _mm256_load_pd, _mm256_add_pd, _mm256_set1_pd, _mm256_setzero_pd, dcossim_reduce_aligned_avx256, hsum_double_avx, _mm256_mul_pd,_mm256_fmadd_pd)
__COSSIM_FUNC(float, __m256, _mm256_loadu_ps, _mm256_add_ps, _mm256_set1_ps, _mm256_setzero_ps, fcossim_reduce_unaligned_avx256, broadcast_reduce_add_si256_psf, _mm256_mul_ps,_mm256_fmadd_ps)
__COSSIM_FUNC(float, __m256, _mm256_load_ps, _mm256_add_ps, _mm256_set1_ps, _mm256_setzero_ps, fcossim_reduce_aligned_avx256, broadcast_reduce_add_si256_psf, _mm256_mul_ps,_mm256_fmadd_ps)
#endif
#ifdef __SSE2__
__D_BSIM_FUNC(double, __m128d, _mm_loadu_pd, _mm_add_pd, _mm_set1_pd, Sleef_sqrtd2_u35, _mm_setzero_pd, dbhattd_reduce_unaligned_sse2, _mm_reduce_add_pdd, _mm_mul_pd, _mm_fmadd_pd)
__D_BSIM_FUNC(double, __m128d, _mm_load_pd, _mm_add_pd, _mm_set1_pd, Sleef_sqrtd2_u35, _mm_setzero_pd, dbhattd_reduce_aligned_sse2, _mm_reduce_add_pdd, _mm_mul_pd, _mm_fmadd_pd)
__D_BSIM_FUNC(float, __m128, _mm_loadu_ps, _mm_add_ps, _mm_set1_ps, Sleef_sqrtf4_u35, _mm_setzero_ps, fbhattd_reduce_unaligned_sse2, _mm_reduce_add_psf, _mm_mul_ps, _mm_fmadd_ps)
__D_BSIM_FUNC(float, __m128, _mm_load_ps, _mm_add_ps, _mm_set1_ps, Sleef_sqrtf4_u35, _mm_setzero_ps, fbhattd_reduce_aligned_sse2, _mm_reduce_add_psf, _mm_mul_ps, _mm_fmadd_ps)
__HELLDIST_FUNC(double, __m128d, _mm_loadu_pd, _mm_add_pd, _mm_set1_pd, Sleef_sqrtd2_u35, _mm_setzero_pd, dhelld_reduce_unaligned_sse2, _mm_reduce_add_pdd, _mm_mul_pd, _mm_sub_pd, _mm_fmadd_pd)
__HELLDIST_FUNC(double, __m128d, _mm_load_pd, _mm_add_pd, _mm_set1_pd, Sleef_sqrtd2_u35, _mm_setzero_pd, dhelld_reduce_aligned_sse2, _mm_reduce_add_pdd, _mm_mul_pd, _mm_sub_pd, _mm_fmadd_pd)
__HELLDIST_FUNC(float, __m128, _mm_loadu_ps, _mm_add_ps, _mm_set1_ps, Sleef_sqrtf4_u35, _mm_setzero_ps, fhelld_reduce_unaligned_sse2, _mm_reduce_add_psf, _mm_mul_ps, _mm_sub_ps, _mm_fmadd_ps)
__HELLDIST_FUNC(float, __m128, _mm_load_ps, _mm_add_ps, _mm_set1_ps, Sleef_sqrtf4_u35, _mm_setzero_ps, fhelld_reduce_aligned_sse2, _mm_reduce_add_psf, _mm_mul_ps, _mm_sub_ps, _mm_fmadd_ps)
__SQRL2_FUNC(double, __m128d, _mm_loadu_pd, _mm_add_pd, _mm_set1_pd, Sleef_sqrtd2_u35, _mm_setzero_pd, dsqrl2_reduce_unaligned_sse2, _mm_reduce_add_pdd, _mm_mul_pd, _mm_sub_pd, _mm_fmadd_pd)
__SQRL2_FUNC(double, __m128d, _mm_load_pd, _mm_add_pd, _mm_set1_pd, Sleef_sqrtd2_u35, _mm_setzero_pd, dsqrl2_reduce_aligned_sse2, _mm_reduce_add_pdd, _mm_mul_pd, _mm_sub_pd, _mm_fmadd_pd)
__SQRL2_FUNC(float, __m128, _mm_loadu_ps, _mm_add_ps, _mm_set1_ps, Sleef_sqrtf4_u35, _mm_setzero_ps, fsqrl2_reduce_unaligned_sse2, _mm_reduce_add_psf, _mm_mul_ps, _mm_sub_ps, _mm_fmadd_ps)
__SQRL2_FUNC(float, __m128, _mm_load_ps, _mm_add_ps, _mm_set1_ps, Sleef_sqrtf4_u35, _mm_setzero_ps, fsqrl2_reduce_aligned_sse2, _mm_reduce_add_psf, _mm_mul_ps, _mm_sub_ps, _mm_fmadd_ps)
__SQRL2_FUNC_NONORM(double, __m128d, _mm_loadu_pd, _mm_add_pd, _mm_set1_pd, Sleef_sqrtd2_u35, _mm_setzero_pd, dnnsqrl2_reduce_unaligned_sse2, _mm_reduce_add_pdd, _mm_mul_pd, _mm_sub_pd, _mm_fmadd_pd)
__SQRL2_FUNC_NONORM(double, __m128d, _mm_load_pd, _mm_add_pd, _mm_set1_pd, Sleef_sqrtd2_u35, _mm_setzero_pd, dnnsqrl2_reduce_aligned_sse2, _mm_reduce_add_pdd, _mm_mul_pd, _mm_sub_pd, _mm_fmadd_pd)
__SQRL2_FUNC_NONORM(float, __m128, _mm_loadu_ps, _mm_add_ps, _mm_set1_ps, Sleef_sqrtf4_u35, _mm_setzero_ps, fnnsqrl2_reduce_unaligned_sse2, _mm_reduce_add_psf, _mm_mul_ps, _mm_sub_ps, _mm_fmadd_ps)
__SQRL2_FUNC_NONORM(float, __m128, _mm_load_ps, _mm_add_ps, _mm_set1_ps, Sleef_sqrtf4_u35, _mm_setzero_ps, fnnsqrl2_reduce_aligned_sse2, _mm_reduce_add_psf, _mm_mul_ps, _mm_sub_ps, _mm_fmadd_ps)
__TVD_FUNC(double, __m128d, _mm_loadu_pd, _mm_add_pd, _mm_set1_pd, Sleef_sqrtd2_u35, _mm_setzero_pd, dtvd_reduce_unaligned_sse2, _mm_reduce_add_pdd, _mm_mul_pd, _mm_sub_pd, _mm_kl_abs_pd, _mm_fmadd_pd)
__TVD_FUNC(double, __m128d, _mm_load_pd, _mm_add_pd, _mm_set1_pd, Sleef_sqrtd2_u35, _mm_setzero_pd, dtvd_reduce_aligned_sse2, _mm_reduce_add_pdd, _mm_mul_pd, _mm_sub_pd, _mm_kl_abs_pd, _mm_fmadd_pd)
__TVD_FUNC(float, __m128, _mm_loadu_ps, _mm_add_ps, _mm_set1_ps, Sleef_sqrtf4_u35, _mm_setzero_ps, ftvd_reduce_unaligned_sse2, _mm_reduce_add_psf, _mm_mul_ps, _mm_sub_ps, _mm_kl_abs_ps, _mm_fmadd_ps)
__TVD_FUNC(float, __m128, _mm_load_ps, _mm_add_ps, _mm_set1_ps, Sleef_sqrtf4_u35, _mm_setzero_ps, ftvd_reduce_aligned_sse2, _mm_reduce_add_psf, _mm_mul_ps, _mm_sub_ps, _mm_kl_abs_ps, _mm_fmadd_ps)
__COSSIM_FUNC(double, __m128d, _mm_loadu_pd, _mm_add_pd, _mm_set1_pd, _mm_setzero_pd, dcossim_reduce_unaligned_sse2, _mm_reduce_add_pdd, _mm_mul_pd, _mm_fmadd_pd)
__COSSIM_FUNC(double, __m128d, _mm_load_pd, _mm_add_pd, _mm_set1_pd, _mm_setzero_pd, dcossim_reduce_aligned_sse2, _mm_reduce_add_pdd, _mm_mul_pd, _mm_fmadd_pd)
__COSSIM_FUNC(float, __m128, _mm_loadu_ps, _mm_add_ps, _mm_set1_ps, _mm_setzero_ps, fcossim_reduce_unaligned_sse2, _mm_reduce_add_psf, _mm_mul_ps, _mm_fmadd_ps)
__COSSIM_FUNC(float, __m128, _mm_load_ps, _mm_add_ps, _mm_set1_ps, _mm_setzero_ps, fcossim_reduce_aligned_sse2, _mm_reduce_add_psf, _mm_mul_ps, _mm_fmadd_ps)
#endif



LIBKL_API double cossim_reduce_aligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhmul, double rhmul, double lhi, double rhi) {
#if __AVX512F__
    return dcossim_reduce_aligned_avx512(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __AVX2__
    return dcossim_reduce_aligned_avx256(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __SSE2__
    return dcossim_reduce_aligned_sse2(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#else
    double ret = 0.;
    for(size_t i = 0; i < n; ++i) ret += sqrt((lhs[i] * lhmul + lhi) * (rhs[i] * rhmul + rhi));
    return ret;
#endif
}
LIBKL_API double cossim_reduce_aligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhmul, float rhmul, float lhi, float rhi) {
#if __AVX512F__
    return fcossim_reduce_aligned_avx512(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __AVX2__
    return fcossim_reduce_aligned_avx256(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __SSE2__
    return fcossim_reduce_aligned_sse2(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#else
    double ret = 0.;
    for(size_t i = 0; i < n; ++i) ret += sqrt((lhs[i] * lhmul + lhi) * (rhs[i] * rhmul + rhi));
    return ret;
#endif
}
LIBKL_API double cossim_reduce_unaligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhmul, double rhmul, double lhi, double rhi) {
#if __AVX512F__
    return dcossim_reduce_unaligned_avx512(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __AVX2__
    return dcossim_reduce_unaligned_avx256(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __SSE2__
    return dcossim_reduce_unaligned_sse2(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#else
    double ret = 0.;
    for(size_t i = 0; i < n; ++i) ret += sqrt((lhs[i] * lhmul + lhi) * (rhs[i] * rhmul + rhi));
    return ret;
#endif
}
LIBKL_API double cossim_reduce_unaligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhmul, float rhmul, float lhi, float rhi) {
#if __AVX512F__
    return fcossim_reduce_unaligned_avx512(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __AVX2__
   return fcossim_reduce_unaligned_avx256(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __SSE2__
    return fcossim_reduce_unaligned_sse2(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#else
    double ret = 0.;
    for(size_t i = 0; i < n; ++i) ret += sqrt((lhs[i] * lhmul + lhi) * (rhs[i] * rhmul + rhi));
    return ret;
#endif
}

LIBKL_API double bhattd_reduce_aligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhmul, double rhmul, double lhi, double rhi) {
#if __AVX512F__
    return dbhattd_reduce_aligned_avx512(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __AVX2__
    return dbhattd_reduce_aligned_avx256(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __SSE2__
    return dbhattd_reduce_aligned_sse2(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#else
    double ret = 0.;
    for(size_t i = 0; i < n; ++i) ret += sqrt((lhs[i] * lhmul + lhi) * (rhs[i] * rhmul + rhi));
    return ret;
#endif
}
LIBKL_API double bhattd_reduce_aligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhmul, float rhmul, float lhi, float rhi) {
#if __AVX512F__
    return fbhattd_reduce_aligned_avx512(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __AVX2__
    return fbhattd_reduce_aligned_avx256(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __SSE2__
    return fbhattd_reduce_aligned_sse2(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#else
    double ret = 0.;
    for(size_t i = 0; i < n; ++i) ret += sqrt((lhs[i] * lhmul + lhi) * (rhs[i] * rhmul + rhi));
    return ret;
#endif
}
LIBKL_API double bhattd_reduce_unaligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhmul, double rhmul, double lhi, double rhi) {
#if __AVX512F__
    return dbhattd_reduce_unaligned_avx512(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __AVX2__
    return dbhattd_reduce_unaligned_avx256(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __SSE2__
    return dbhattd_reduce_unaligned_sse2(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#else
    double ret = 0.;
    for(size_t i = 0; i < n; ++i) ret += sqrt((lhs[i] * lhmul + lhi) * (rhs[i] * rhmul + rhi));
    return ret;
#endif
}
LIBKL_API double bhattd_reduce_unaligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhmul, float rhmul, float lhi, float rhi) {
#if __AVX512F__
    return fbhattd_reduce_unaligned_avx512(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __AVX2__
   return fbhattd_reduce_unaligned_avx256(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __SSE2__
    return fbhattd_reduce_unaligned_sse2(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#else
    double ret = 0.;
    for(size_t i = 0; i < n; ++i) ret += sqrt((lhs[i] * lhmul + lhi) * (rhs[i] * rhmul + rhi));
    return ret;
#endif
}
LIBKL_API double helld_reduce_aligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhmul, double rhmul, double lhi, double rhi) {
#if __AVX512F__
    return dhelld_reduce_aligned_avx512(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __AVX2__
    return dhelld_reduce_aligned_avx256(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __SSE2__
    return dhelld_reduce_aligned_sse2(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#else
    double ret = 0.;
    for(size_t i = 0; i < n; ++i) {
        double v = sqrt(lhs[i] * lhmul + lhi) - sqrt(rhs[i] * rhmul + rhi);
        ret += v * v;
    }
    return ret;
#endif
}
LIBKL_API double helld_reduce_aligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhmul, float rhmul, float lhi, float rhi) {
#if __AVX512F__
    return fhelld_reduce_aligned_avx512(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __AVX2__
    return fhelld_reduce_aligned_avx256(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __SSE2__
    return fhelld_reduce_aligned_sse2(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#else
    double ret = 0.;
    for(size_t i = 0; i < n; ++i) {
        double v = sqrt(lhs[i] * lhmul + lhi) - sqrt(rhs[i] * rhmul + rhi);
        ret += v * v;
    }
    return ret;
#endif
}
LIBKL_API double helld_reduce_unaligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhmul, double rhmul, double lhi, double rhi) {
#if __AVX512F__
    return dhelld_reduce_unaligned_avx512(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __AVX2__
    return dhelld_reduce_unaligned_avx256(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __SSE2__
    return dhelld_reduce_unaligned_sse2(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#else
    double ret = 0.;
    for(size_t i = 0; i < n; ++i) {
        double v = sqrt(lhs[i] * lhmul + lhi) - sqrt(rhs[i] * rhmul + rhi);
        ret += v * v;
    }
    return ret;
#endif
}
LIBKL_API double helld_reduce_unaligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhmul, float rhmul, float lhi, float rhi) {
#if __AVX512F__
    return fhelld_reduce_unaligned_avx512(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __AVX2__
    return fhelld_reduce_unaligned_avx256(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __SSE2__
    return fhelld_reduce_unaligned_sse2(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#else
    double ret = 0.;
    for(size_t i = 0; i < n; ++i) {
        double v = sqrt(lhs[i] * lhmul + lhi) - sqrt(rhs[i] * rhmul + rhi);
        ret += v * v;
    }
    return ret;
#endif
}
LIBKL_API double nnsqrl2_reduce_unaligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n) {
#if __AVX512F__
    return fnnsqrl2_reduce_unaligned_avx512(lhs, rhs, n);
#elif __AVX2__
    return fnnsqrl2_reduce_unaligned_avx256(lhs, rhs, n);
#elif __SSE2__
    return fnnsqrl2_reduce_unaligned_sse2(lhs, rhs, n);
#else
    float ret = 0.;
    for(size_t i = 0; i < n; ++i) {
        float v = lhs[i] - rhs[i];
        ret += v * v;
    }
    return ret;
#endif
}

LIBKL_API double nnsqrl2_reduce_aligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n) {
#if __AVX512F__
    return fnnsqrl2_reduce_aligned_avx512(lhs, rhs, n);
#elif __AVX2__
    return fnnsqrl2_reduce_aligned_avx256(lhs, rhs, n);
#elif __SSE2__
    return fnnsqrl2_reduce_aligned_sse2(lhs, rhs, n);
#else
    float ret = 0.;
    for(size_t i = 0; i < n; ++i) {
        float v = lhs[i] - rhs[i];
        ret += v * v;
    }
    return ret;
#endif
}

LIBKL_API double nnsqrl2_reduce_unaligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n) {
#if __AVX512F__
    return dnnsqrl2_reduce_unaligned_avx512(lhs, rhs, n);
#elif __AVX2__
    return dnnsqrl2_reduce_unaligned_avx256(lhs, rhs, n);
#elif __SSE2__
    return dnnsqrl2_reduce_unaligned_sse2(lhs, rhs, n);
#else
    double ret = 0.;
    for(size_t i = 0; i < n; ++i) {
        double v = lhs[i] - rhs[i];
        ret += v * v;
    }
    return ret;
#endif
}
LIBKL_API double nnsqrl2_reduce_aligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n) {
#if __AVX512F__
    return dnnsqrl2_reduce_aligned_avx512(lhs, rhs, n);
#elif __AVX2__
    return dnnsqrl2_reduce_aligned_avx256(lhs, rhs, n);
#elif __SSE2__
    return dnnsqrl2_reduce_aligned_sse2(lhs, rhs, n);
#else
    double ret = 0.;
    for(size_t i = 0; i < n; ++i) {
        double v = lhs[i] - rhs[i];
        ret += v * v;
    }
    return ret;
#endif
}
LIBKL_API double sqrl2_reduce_aligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhmul, double rhmul, double lhi, double rhi) {
#if __AVX512F__
    return dsqrl2_reduce_aligned_avx512(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __AVX2__
    return dsqrl2_reduce_aligned_avx256(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __SSE2__
    return dsqrl2_reduce_aligned_sse2(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#else
    double ret = 0.;
    for(size_t i = 0; i < n; ++i) {
        double v = (lhs[i] * lhmul + lhi) - (rhs[i] * rhmul + rhi);
        ret += v * v;
    }
    return ret;
#endif
}
LIBKL_API double sqrl2_reduce_aligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhmul, float rhmul, float lhi, float rhi) {
#if __AVX512F__
   return fsqrl2_reduce_aligned_avx512(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __AVX2__
    return fsqrl2_reduce_aligned_avx256(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __SSE2__
    return fsqrl2_reduce_aligned_sse2(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#else
    double ret = 0.;
    for(size_t i = 0; i < n; ++i) {
        double v = (lhs[i] * lhmul + lhi) - (rhs[i] * rhmul + rhi);
        ret += v * v;
    }
    return ret;
#endif
}
LIBKL_API double sqrl2_reduce_unaligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhmul, double rhmul, double lhi, double rhi) {
#if __AVX512F__
    return dsqrl2_reduce_unaligned_avx512(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __AVX2__
    return dsqrl2_reduce_unaligned_avx256(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __SSE2__
    return dsqrl2_reduce_unaligned_sse2(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#else
    double ret = 0.;
    for(size_t i = 0; i < n; ++i) {
        double v = (lhs[i] * lhmul + lhi) - (rhs[i] * rhmul + rhi);
        ret += v * v;
    }
    return ret;
#endif
}
LIBKL_API double sqrl2_reduce_unaligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhmul, float rhmul, float lhi, float rhi) {
#if __AVX512F__
    return fsqrl2_reduce_unaligned_avx512(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __AVX2__
    return fsqrl2_reduce_unaligned_avx256(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __SSE2__
    return fsqrl2_reduce_unaligned_sse2(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#else
    double ret = 0.;
    for(size_t i = 0; i < n; ++i) {
        double v = (lhs[i] * lhmul + lhi) - (rhs[i] * rhmul + rhi);
        ret += v * v;
    }
    return ret;
#endif
}

LIBKL_API double tvd_reduce_aligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhmul, double rhmul, double lhi, double rhi) {
#if __AVX512F__
    return dtvd_reduce_aligned_avx512(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __AVX2__
    return dtvd_reduce_aligned_avx256(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __SSE2__
    return dtvd_reduce_aligned_sse2(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#else
    double ret = 0.;
    for(size_t i = 0; i < n; ++i) {
        double v = (lhs[i] * lhmul + lhi) - (rhs[i] * rhmul + rhi);
        ret += fabs(v);
    }
    return ret * .5;
#endif
}
LIBKL_API double tvd_reduce_aligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhmul, float rhmul, float lhi, float rhi) {
#if __AVX512F__
   return ftvd_reduce_aligned_avx512(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __AVX2__
    return ftvd_reduce_aligned_avx256(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __SSE2__
    return ftvd_reduce_aligned_sse2(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#else
    double ret = 0.;
    for(size_t i = 0; i < n; ++i) {
        double v = (lhs[i] * lhmul + lhi) - (rhs[i] * rhmul + rhi);
        ret += fabs(v);
    }
    return ret * .5;
#endif
}
LIBKL_API double tvd_reduce_unaligned_d(const double *const __restrict__ lhs, const double *const __restrict__ rhs, const size_t n, double lhmul, double rhmul, double lhi, double rhi) {
#if __AVX512F__
    return dtvd_reduce_unaligned_avx512(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __AVX2__
    return dtvd_reduce_unaligned_avx256(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __SSE2__
    return dtvd_reduce_unaligned_sse2(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#else
    double ret = 0.;
    for(size_t i = 0; i < n; ++i) {
        double v = (lhs[i] * lhmul + lhi) - (rhs[i] * rhmul + rhi);
        ret += fabs(v);
    }
    return ret * .5;
#endif
}
LIBKL_API double tvd_reduce_unaligned_f(const float *const __restrict__ lhs, const float *const __restrict__ rhs, const size_t n, float lhmul, float rhmul, float lhi, float rhi) {
#if __AVX512F__
    return ftvd_reduce_unaligned_avx512(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __AVX2__
    return ftvd_reduce_unaligned_avx256(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#elif __SSE2__
    return ftvd_reduce_unaligned_sse2(lhs, rhs, n, lhmul, rhmul, lhi, rhi);
#else
    double ret = 0.;
    for(size_t i = 0; i < n; ++i) {
        double v = (lhs[i] * lhmul + lhi) - (rhs[i] * rhmul + rhi);
        ret += fabs(v);
    }
    return ret * .5;
#endif
}

#undef __D_BSIM_FUNC
#undef __HELLDIST_FUNC
#undef __TVD_FUNC


#ifdef LIBKL_HIGH_PRECISION
#undef Sleef_logd2_u35
#undef Sleef_logd4_u35
#undef Sleef_logd8_u35
#undef Sleef_logf4_u35
#undef Sleef_logf8_u35
#undef Sleef_logd2_u35
#undef Sleef_sqrtd4_u35
#undef Sleef_sqrtd8_u35
#undef Sleef_sqrtf4_u35
#undef Sleef_sqrtf8_u35
#undef Sleef_sqrtf16_u35
#endif


#ifdef __cplusplus
}
#endif
