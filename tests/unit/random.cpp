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
    using simd_mask_t = sve::experimental::simd_mask<T>;

    auto x1 = simd_t(sve_impl::simd_impl<T>::index0123) + T(1);
    std::cout << x1 << '\n';
    auto m1 = simd_mask_t(false);
    std::cout << m1 << '\n';
    for (int i = 0; i < x1.size(); i += 3)
    {
        m1.set(i, true);
    }
    std::cout << m1 << '\n';
    if constexpr(sizeof(T) >= 4)
    {
        std::cout << compact(m1, x1) << '\n';
    }
    std::cout << choose(m1, T(5) * x1, -x1) << '\n';
    mask_assign(!m1, x1, -x1);
    std::cout << x1 << '\n';
    std::cout << '\n';
}

int main()
{
    // test<uint8_t>();
    test<int8_t>();
    test<uint16_t>();
    test<int16_t>();
    test<uint32_t>();
    test<int32_t>();
    test<float16_t>();
    test<float32_t>();
    test<float64_t>();
}