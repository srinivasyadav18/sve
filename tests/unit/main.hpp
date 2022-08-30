template <typename T>
void test(std::size_t seed = 42)
{
    int len = (SVE_LEN / 8) / sizeof(T);
    std::vector<T> input(len), 
                    seq_output(len),
                    sve_simd_output(len),
                    acle_output(len);
    std::generate(input.begin(), input.end(), random_gen<T>{seed});

    seq(input, seq_output);
    sve_simd(input, sve_simd_output);
    acle(input, acle_output);

#if defined(DEBUG)
        print_vector(input);
        print_vector(seq_output);
        print_vector(sve_simd_output);
        print_vector(acle_output);
        std::cout << '\n';
#endif

    assert(input == seq_output);
    assert(seq_output == sve_simd_output);
    assert(sve_simd_output == acle_output);
}

int main()
{
    for (int i = 0; i < 100; i++)
    {
        test<int>(true_random());
        test<float>(true_random());
        test<double>(true_random());
    }
    return 0;
}