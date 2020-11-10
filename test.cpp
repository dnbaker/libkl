#include "libkl.h"
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <iterator>
using std::size_t;

//#define T T
template<typename T>
int main_fn() {
    T *v1 = 0, *v2 = 0, *v3 = 0;
    size_t nelem = 20;
    posix_memalign((void **)&v1, 64, sizeof(*v1) * nelem);
    posix_memalign((void **)&v2, 64, sizeof(*v2) * nelem);
    posix_memalign((void **)&v3, 64, sizeof(*v3) * nelem);
    T vs1 = 0., vs2 = 0., vs3 = 0.;
    for(size_t i = 0; i < nelem; ++i) {
        v1[i] = i;
        v2[i] = i;
        vs1 += v1[i];
        vs2 += v2[i];
    }
    std::copy(std::reverse_iterator<T *>(v1 + nelem), std::reverse_iterator<T *>(v1),
              v3);
    T prior = 1.;
    T psum = prior * nelem;
    T lambda = (vs1 + psum) / (vs1 + vs2 + psum * 2.f);
    T m1v = 1. - lambda;
    for(size_t i = 0; i < nelem; ++i) {
        v2[i] /= (vs2 + psum);
        v1[i] /= (vs1 + psum);
    }
    auto v1sum = std::accumulate(v1, v1 + nelem, 0.);
    auto v2sum = std::accumulate(v2, v2 + nelem, 0.);
    T lhi = prior / (vs1 + psum), rhi = prior / (vs2 + psum);
    //fprintf(stderr, "sums after norm: %g, %g. With psum: %g, %g\n", v1sum, v2sum, v1sum + lhi * nelem, v2sum + rhi * nelem);
    auto start = std::chrono::high_resolution_clock::now();
    //fprintf(stderr, "lhi: %g: rhi: %g.\n", lhi, rhi);
    T v11_man = 0.;
    for(size_t i = 0; i < nelem; ++i) {
        auto xv = v1[i] + lhi, yv = v2[i] + rhi;
        auto mnv = (xv * lambda  + yv * m1v);
        auto xc = xv * std::log(xv / mnv);
        auto yc = yv * std::log(yv / mnv);
        auto c = lambda * xc + m1v * yc;
        //fprintf(stderr, "mnv: %g. x: %g. y: %g\n", mnv, v1[i], v2[i]);
        //fprintf(stderr, "xc: %g. yc: %g\n", xc, yc);
        v11_man += c;
        //std::fprintf(stderr, "%g %g -> %g inc, csum %g\n", v1[i], v2[i], c, v11_man);
    }
    //std::fprintf(stderr, "manual: %g\n", v11_man);
    T v11 = __llr_reduce_aligned(v1, v1, nelem, lambda, lhi, rhi);
    T v22 = __llr_reduce_aligned(v2, v2, nelem, lambda, lhi, rhi);
    T v12 = __llr_reduce_aligned(v1, v2, nelem, lambda, lhi, rhi);
    assert(v11 == 0.);
    assert(v22 == 0.);
    assert(std::abs(v12 - 0.) < 1e-10);
    assert(std::equal(v1, v1 + nelem, v2));
    assert(lhi == rhi);
    v11 = __kl_reduce_aligned(v1, v1, nelem, lhi, rhi);
    v22 = __kl_reduce_aligned(v2, v2, nelem, lhi, rhi);
    v12 = __kl_reduce_aligned(v1, v2, nelem, lhi, rhi);
    assert(v11 == 0. || !std::fprintf(stderr, "v1 and v1 -> %g\n", v11));
    assert(v22 == 0. || !std::fprintf(stderr, "v2 and v2 -> %g\n", v22));
    assert(v12 == 0.);
    T s = 0.;
    for(size_t i = 0; i < nelem; ++i) {
        auto lhv = v1[i] + lhi, rhv = v2[i] + rhi;
        auto mnv = (lhv + rhv) / 2;
        s += lambda * lhv * std::log(lhv / mnv) + m1v * rhv * std::log(rhv / mnv);
    }
    assert(s == 0.);
// Next, test scaling
    std::transform(v1, v1 + nelem, v1, [](auto x) {return 2 * x + 1;});
    std::transform(v2, v2 + nelem, v2, [](auto x) {return 2 * x + 1;});
    T v1s = std::accumulate(v1, v1 + nelem, 0.),
           v2s = std::accumulate(v2, v2 + nelem, 0.);
    std::transform(v1, v1 + nelem, v1, [v1s](auto x) {return x / v1s;});
    std::transform(v2, v2 + nelem, v2, [v2s](auto x) {return x / v2s;});
    lambda = (std::accumulate(v1, v1 + nelem, 0.) / (std::accumulate(v1, v1 + nelem, 0.) + std::accumulate(v2, v2 + nelem, 0.)));
    assert(__llr_reduce_aligned(v1, v2, nelem, lambda, 0., 0.) == 0.);

    fprintf(stderr, "Testing scaling\n");
    for(size_t i = 0; i < nelem; ++i) {
        v1[i] = i;
        v2[nelem - i - 1] = i;
    }
    v1s = std::accumulate(v1, v1 + nelem, 0.) + 1. * nelem;
    v2s = std::accumulate(v2, v2 + nelem, 0.) + 1. * nelem;
    lhi = 1. / (v1s), rhi = 1. / (v1s);
    std::transform(v1, v1 + nelem, v1, [v1s](auto x) {return x / v1s;});
    std::transform(v2, v2 + nelem, v2, [v2s](auto x) {return x / v2s;});
    v1s = std::accumulate(v1, v1 + nelem, 0.),
    v2s = std::accumulate(v2, v2 + nelem, 0.);
    fprintf(stderr, "Testing aligned kl reduction. sum lhs: %g. sum rhs: %g. (%g/%g)\n", v1s, v2s, v1s + lhi * nelem, v2s + rhi * nelem);
    lambda = (v2s + nelem) / (v2s + v1s + nelem * 2);
    v12 = __llr_reduce_aligned(v1, v2, nelem, lambda, lhi, rhi);
    fprintf(stderr, "manual via simd: %g. via numpy: %g\n", v12, 0.1705398845054989);
    assert(std::abs(v12) - 0.3410797690109978 < 1e-6);
    v12 = __kl_reduce_aligned(v1, v2, nelem, lhi, rhi);
    fprintf(stderr, "manual via simd: %g. via numpy: %g\n", v12, 0.8102686372626047);
    assert(std::abs(v12) - 0.8102686372626047 < 1e-6);


    //std::fprintf(stderr, "Manual value: %g. Value via __llr_reduce_aligned: %g\n", s, v12);
    auto stop = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[%s] Finished \n", __PRETTY_FUNCTION__);
    return 0;
}

int main() {
    return main_fn<float>() | main_fn<double>();
}
