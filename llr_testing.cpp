#include "libkl.h"
#include <cstdio>
#include <chrono>
#include <cstdlib>
using std::size_t;

using namespace libkl;
int main() {
    float *v1 = 0, *v2 = 0, *v3 = 0, *v4 = 0;
    size_t nelem = 100000;
    if(const char *s = std::getenv("NELEM"))
        nelem = std::strtoull(s, nullptr, 10);
    posix_memalign((void **)&v1, 64, sizeof(*v1) * nelem);
    posix_memalign((void **)&v2, 64, sizeof(*v2) * nelem);
    posix_memalign((void **)&v3, 64, sizeof(*v3) * nelem);
    posix_memalign((void **)&v4, 64, sizeof(*v4) * nelem);
    double vs1 = 0., vs2 = 0., vs3 = 0., vs4 = 0.;
    for(size_t i = 0; i < nelem; ++i) {
        v1[i] = std::rand() % 64 + 0.1;
        v2[i] = std::rand() % 64 + 0.1;
        v3[i] = std::rand() % 64 + 0.1;
        v4[i] = std::rand() % 64 + 0.1;
        vs1 += v1[i];
        vs2 += v2[i];
        vs3 += v3[i];
        vs4 += v4[i];
    }
    for(size_t i = 0; i < nelem; ++i) {
        v4[i] /= vs4;
        v3[i] /= vs3;
        v2[i] /= vs2;
        v1[i] /= vs1;
    }
    auto lhi = .1 / vs1, rhi = .1 / vs2;
    auto lambda = vs1 / (vs1 + vs2);
    auto start = std::chrono::high_resolution_clock::now();
    double s = 0.;
    for(size_t k = 0; k < 10; ++k) {
        s += llr_reduce_aligned(v1, v2, nelem, lambda, lhi, rhi);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "Strategy took %gms for 10 trials\n", std::chrono::duration<double, std::nano>(stop - start).count());
}
