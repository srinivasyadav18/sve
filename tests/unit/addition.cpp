#include <sve/sve.hpp>
#include <iostream>

int main()
{
    using namespace sve::experimental;
    simd<float> x(5);
    std::cout << "xlen : " << x.size() << '\n';
    std::cout << "x : " << x << '\n';
    return 0;
}
