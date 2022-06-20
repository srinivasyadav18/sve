#include <sve/sve.hpp>
#include <iostream>
#include <vector>
#include <numeric>

void seq()
{
    typedef svint32_t Vector __attribute__((arm_sve_vector_bits(SVE_LEN)));
    Vector vec = svdup_s32(5);
    std::cout << "addv : " << svaddv(svptrue_b32(), vec) << "\n";

    for (int idx = 0; idx < 4; idx++)
    {
        auto index = svptrue_pat_b32(SV_VL1);
        for (int i = 0; i < idx; i++)
        {
            index = svpnext_b32(svptrue_b32(), index);
        }
        std::cout << "idx : " << idx << " " <<  svlastb(index, vec) << "\n";
    }
}

template <typename T>
void test()
{
    using namespace sve::experimental;
    simd<T> x(5);
    std::cout << "addv : " << x.addv() << "\n";
    std::cout << "xlen : " << x.size() << '\n';
    std::cout << "x : " << x << '\n';
    auto tmp = x + 1;
    std::cout << "got x + 1 into tmp" << '\n';
    std::cout << "x : " << tmp << '\n';
    std::cout << "x : " << x++ << '\n';
    std::cout << "x : " << ++x << "\n\n";

    std::vector<T> data(x.size());
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
}

int main()
{
    seq();
    test<int>();
    test<float>();
    test<double>();
    test<int8_t>();
    test<int16_t>();
    return 0;
}

