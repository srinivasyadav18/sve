#include <sve/sve.hpp>
#include <iostream>
#include <vector>
#include <numeric>

int main()
{
    using namespace sve::experimental;
    simd<float> x(5);
    std::cout << "xlen : " << x.size() << '\n';
    std::cout << "x : " << x << '\n';
    std::cout << "x : " << x + 1 << '\n';
    std::cout << "x : " << x++ << '\n';
    std::cout << "x : " << ++x << "\n\n";

    std::vector<float> data(x.size());
    std::iota(data.begin(), data.end(), 1);
    x.copy_from(data.data(), vector_aligned);

    std::cout << "x : " << x << '\n';
    std::cout << "x : " << x + 1 << '\n';
    std::cout << "x : " << x++ << '\n';
    std::cout << "x : " << ++x << "\n\n";
    x.copy_from(data.data(), vector_aligned);

    std::cout << "x : " << x << '\n';
    std::cout << "x : " << x + 1 << '\n';
    std::cout << "x : " << x++ << '\n';
    std::cout << "x : " << ++x << "\n\n";

    std::cout << "x : " << x << '\n';
    std::cout << "x : " << x - 1 << '\n';
    std::cout << "x : " << x-- << '\n';
    std::cout << "x : " << --x << '\n';
    return 0;
}
