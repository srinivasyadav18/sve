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

    simd_t x1 = simd_t(sve_impl::simd_impl<T>::index0123);
    auto x2 = reduce(x1, std::plus<>{});
    std::cout << x1 << '\n';
    std::cout << x2 << '\n';
    std::cout << x1.addv() << '\n';
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
