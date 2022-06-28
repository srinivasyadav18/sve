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
    auto x1 = simd_t(sve_impl::simd_impl<T>::index0123);
    std::cout << x1 << '\n';
    auto x2 = x1.size();
    assert(((x2-1)*(x2/2)) == x1.addv());
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