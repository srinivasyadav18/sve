#include <sve/sve.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>

template <typename T>
void test()
{
    using namespace sve::experimental;
    using simd_t = sve::experimental::simd<T>;

    std::vector<T> r1(simd_t::size());
    std::iota(r1.begin(), r1.end(), 0);
    auto r2 = r1, r3 = r1;
    std::inclusive_scan(r1.begin(), r1.end(), r2.begin());
    std::exclusive_scan(r1.begin(), r1.end(), r3.begin(), 0);

    simd_t x1 = simd_t(sve_impl::simd_impl<T>::index0123);
    simd_t x2 = inclusive_scan(x1, [](auto a, auto b){return a + b;});
    simd_t x3 = exclusive_scan(x1, [](auto a, auto b){return a + b;}, T(0));

    std::cout << x1 << '\n';
    std::cout << x2 << '\n';
    std::cout << x3 << "\n\n";

    simd_t x4(r1.data(), sve::experimental::vector_aligned);
    simd_t x5(r2.data(), sve::experimental::vector_aligned);
    simd_t x6(r3.data(), sve::experimental::vector_aligned);

    std::cout << x4 << '\n';
    std::cout << x5 << '\n';
    std::cout << x6 << "\n\n";

    assert((x1 == x4).popcount() == simd_t::size());
    assert((x2 == x5).popcount() == simd_t::size());
    assert((x3 == x6).popcount() == simd_t::size());
}

int main()
{
    test<uint8_t>();
    test<int8_t>();
    test<uint16_t>();
    test<int16_t>();
    test<uint32_t>();
    test<int32_t>();
    test<float16_t>();
    test<float32_t>();
    test<float64_t>();
}
