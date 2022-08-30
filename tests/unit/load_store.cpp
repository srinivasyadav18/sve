#include <sve/sve.hpp>
#include "utility.hpp"

template <typename T>
void seq(std::vector<T> const &input, std::vector<T>& output)
{
    for (int i = 0; i < input.size(); i++)
    {
        T x = input[i];
        output[i] = x;
    }
}

template <typename T>
void sve_simd(std::vector<T> const &input, std::vector<T>& output)
{
    using namespace sve::experimental;
    using simd_t = simd<T>;

    simd_t x(input.data(), vector_aligned);
    x.copy_to(output.data(), vector_aligned);
}

template <typename T>
void acle(std::vector<T> const &input, std::vector<T>& output)
{
    if constexpr(std::is_same_v<T, int>)
    {
        typedef svint32_t Vector __attribute__((arm_sve_vector_bits(SVE_LEN)));
        Vector vec = svld1(svptrue_b32(), input.data());
        svst1(svptrue_b32(), output.data(), vec);
    }
    if constexpr(std::is_same_v<T, float>)
    {
        typedef svfloat32_t Vector __attribute__((arm_sve_vector_bits(SVE_LEN)));
        Vector vec = svld1(svptrue_b32(), input.data());
        svst1(svptrue_b32(), output.data(), vec);
    }
    if constexpr(std::is_same_v<T, double>)
    {
        typedef svfloat64_t Vector __attribute__((arm_sve_vector_bits(SVE_LEN)));
        Vector vec = svld1(svptrue_b64(), input.data());
        svst1(svptrue_b64(), output.data(), vec);
    }
}

#include "main.hpp"
