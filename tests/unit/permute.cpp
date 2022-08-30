#include <sve/sve.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>

template <typename T>
void test()
{
    std::cout << "=========== " << " ===========\n"; 
    using namespace sve::experimental;
    using simd_t = sve::experimental::simd<T>;
    using simd_mask_t = sve::experimental::simd_mask<T>;

    auto x1 = simd_t(sve_impl::simd_impl<T>::index0123) + T(1);
    std::cout << "x = " << x1 << "\n\n";
    std::cout << "interleave even odd\n";
    std::cout << interleave_even(x1, x1) << '\n';
    std::cout << interleave_odd(x1, x1) << '\n';
    std::cout << '\n';

    std::cout << "select even odd\n";
    std::cout << select_even(x1, x1) << '\n';
    std::cout << select_odd(x1, x1) << '\n';
    std::cout << '\n';

    std::cout << "lower upper half\n";
    std::cout << lower_half(x1, x1) << '\n';
    std::cout << upper_half(x1, x1) << '\n';
    std::cout << "\n";

    std::cout << "splice\n";
    std::cout << (x1 < T(4)) << '\n';
    std::cout << (T(2)*x1) << '\n';
    std::cout << (T(3)*x1) << '\n';
    std::cout << splice(x1 < T(4), T(2)*x1, T(3)*x1);
    std::cout << "\n";

    std::cout << "reverse\n";
    std::cout << reverse(x1) << '\n';
    std::cout << "\n";

    std::cout << "fma\n";
    std::cout << fma(x1, simd_t(2), simd_t(3)) << '\n';
    std::cout << "\n";

    if constexpr(std::is_floating_point_v<T>)
    {
        std::cout << "sqrt\n";
        std::cout << sqrt(x1) << '\n';
        std::cout << "\n";
    }
    std::cout << "\n";

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