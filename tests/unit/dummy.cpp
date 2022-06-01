#include <sve/sve.hpp>

int main()
{
    using namespace sve::experimental;
    std::cout << is_simd_v<simd<int>> << '\n';
    std::cout << is_simd_v<simd<float>> << '\n';
    std::cout << is_simd_v<simd<int, simd_abi::native<int>>> << '\n';
    std::cout << is_simd_v<simd<int, simd_abi::compatible<int>>> << '\n';
    std::cout << is_simd_v<simd<int>> << '\n';
    std::cout << is_simd_v<float> << '\n';
    std::cout << is_simd_mask_v<float> << '\n';
    std::cout << is_simd_mask_v<simd_mask<double>> << '\n';
    std::cout << is_simd_mask_v<simd_mask<int>> << '\n';
    std::cout << is_simd_v<simd<int, simd_abi::fixed_size<int, 5>>> << '\n';
    std::cout << is_simd_v<fixed_size_simd<int, 5>> << '\n';

    return 0;
}