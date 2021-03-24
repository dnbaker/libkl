#include "libkl.ho.h"
#include <cassert>
#include <iostream>
#include <vector>
#undef NDEBUG

int main() {
    std::vector<double> lhs(1000), rhs(1000);
    for(size_t i = 0; i < 1000; ++i) lhs[i] = i & 1 ? .7: 1.4;
    for(size_t i = 0; i < 1000; ++i) rhs[i] = i & 1 ? 1.4: .7;
    auto cossim = libkl::cossim_reduce(lhs.data(), rhs.data(), 1000, 1., 1., 0., 0.);
    assert(std::abs(0.8 - cossim) < 1e-4);
}
