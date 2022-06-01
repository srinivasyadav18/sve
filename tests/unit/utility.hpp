#include <random>
#include <chrono>
#include <iostream>
#include <limits>
#include <vector>
#include <cassert>

uint64_t true_random()
{
    using namespace std::chrono;
    return duration_cast<milliseconds>(
            system_clock::now().time_since_epoch()).count();
}

template <typename T>
struct random_gen{};

template <>
struct random_gen<int>
{
    std::mt19937 mersenne_engine;
    std::uniform_int_distribution<int> dist{
            -1024, 
            1024};
    
    random_gen(std::size_t seed)
    {
        mersenne_engine = std::mt19937{seed};
    }
    auto operator()()
    {
        return dist(mersenne_engine);
    }
};

template <>
struct random_gen<float>
{
    std::mt19937 mersenne_engine;
    std::uniform_real_distribution<float> dist{
            -1024, 
            1024};
    
    random_gen(std::size_t seed)
    {
        mersenne_engine = std::mt19937{seed};
    }
    auto operator()()
    {
        return dist(mersenne_engine);
    }
};

template <>
struct random_gen<double>
{
    std::mt19937 mersenne_engine;
    std::uniform_real_distribution<double> dist{
                                        -1024, 
                                        1024};
                        // std::numeric_limits<double>::min(), 
                        // std::numeric_limits<double>::max()};
    
    random_gen(std::size_t seed)
    {
        mersenne_engine = std::mt19937{seed};
    }
    auto operator()()
    {
        return dist(mersenne_engine);
    }
};

template <typename T>
void print_vector(std::vector<T> const& t)
{
    std::cout << "[T = " << typeid(T).name() << " size = " << t.size() << "]\n";
    for (int i = 0; i < t.size(); i++)
    {
        std::cout << t[i] << ' ';
    }
    std::cout << '\n';
}