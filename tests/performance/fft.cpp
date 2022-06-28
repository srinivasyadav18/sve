#include <sve/sve.hpp>
#include <iostream>
#include <vector>
#include <numeric>

#include <complex>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <cassert>
#include <fstream>
#include <chrono>

template <typename T>
inline void complex_mul(T a, T b, T c, T d, T& res_re, T& res_im)
{
    res_re = a*c - b*d;
    res_im = a*d + b*c;
}

template <typename T>
std::vector<T> gen_signal(T freq)
{
    const T PI = acos(-1);
    T sf = T(16) * freq;
    T ts = 1/sf;
    std::vector<T> ret(sf, 0);
    int idx = 0;
    for (T i = T(0); i < T(1); i += ts)
    {
        ret[idx++] = T(10) * sin(T(2.0) * PI * freq * i);
    }
    return ret;
}

template <typename T>
void fft(std::vector<T>& re, std::vector<T>& im, bool invert) {
    const T PI = acos(-1);
    int n = re.size();

    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
        {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }

    for (int len = 2; len <= n; len <<= 1) {
        T ang = 2 * PI / len * (invert ? -1 : 1);
        T wlen_re = cos(ang);
        T wlen_im = sin(ang);
        for (int i = 0; i < n; i += len) {
            T w_re = 1;
            T w_im = 0;
            for (int j = 0; j < len / 2; j++) {
                int u = i+j;
                int v = i+j+len/2;
                T u_re = re[u];
                T u_im = im[u];
                T v_re = re[v];
                T v_im = im[v];
                complex_mul(v_re, v_im, w_re, w_im, v_re, v_im);
                re[u] = u_re + v_re;
                im[u] = u_im + v_im;
                re[v] = u_re - v_re;
                im[v] = u_im - v_im;
                complex_mul(w_re, w_im, wlen_re, wlen_im, w_re, w_im);
            }
        }
    }

    if (invert) {
        for (T & x : re)
            x /= n;
        for (T & x : im)
            x /= n;
    }
}

template <typename T>
void fft_simd(std::vector<T>& re, std::vector<T>& im, bool invert) {
    using simd_t = sve::experimental::simd<T>;
    const T PI = acos(-1);

    int n = re.size();

    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;

        if (i < j)
        {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }

    for (int len = 2; len <= n; len <<= 1) {
        if (len < 2 * (int)simd_t::size())
        {
            T ang = 2 * PI / len * (invert ? -1 : 1);

            T wlen_re = cos(ang);
            T wlen_im = sin(ang);
            for (int i = 0; i < n; i += len) {

                T w_re = 1;
                T w_im = 0;

                for (int j = 0; j < len / 2; j++) {
                    int u = i+j;
                    int v = i+j+len/2;

                    T u_re = re[u];
                    T u_im = im[u];
                    T v_re = re[v];
                    T v_im = im[v];
                    complex_mul(v_re, v_im, w_re, w_im, v_re, v_im);

                    re[u] = u_re + v_re;
                    im[u] = u_im + v_im;

                    re[v] = u_re - v_re;
                    im[v] = u_im - v_im;

                    complex_mul(w_re, w_im, wlen_re, wlen_im, w_re, w_im);
                }
            }
        }
        else
        {
            T ang = 2 * PI / len * (invert ? -1 : 1);
            T cos_ang = cos(ang), sin_ang = sin(ang);
            simd_t wlen_re(cos_ang);
            simd_t wlen_im(sin_ang);
            for (int i = 0; i < n; i += len) 
            {
                std::vector<T> w_re_arr(simd_t::size(), 0), w_im_arr(simd_t::size(), 0);
                w_re_arr[0] = 1;
                w_im_arr[0] = 0;
                for (int k = 1; k < w_im_arr.size(); k++)
                {
                    complex_mul(w_re_arr[k-1], w_im_arr[k-1],cos_ang, sin_ang, w_re_arr[k], w_im_arr[k]);
                }

                simd_t w_re(w_re_arr.data(), sve::experimental::vector_aligned);
                simd_t w_im(w_im_arr.data(), sve::experimental::vector_aligned);
                auto w_re_init = w_re;
                auto w_im_init = w_im;
                for (int j = 0; j < len / 2; j += simd_t::size()) 
                {
                    int u = i+j;
                    int v = i+j+len/2;

                    simd_t u_re(re.data() + u, sve::experimental::vector_aligned);
                    simd_t u_im(im.data() + u, sve::experimental::vector_aligned);
                    simd_t v_re(re.data() + v, sve::experimental::vector_aligned);
                    simd_t v_im(im.data() + v, sve::experimental::vector_aligned);

                    complex_mul(v_re, v_im, w_re, w_im, v_re, v_im);

                    auto uv_re = u_re + v_re;
                    auto uv_im = u_im + v_im;
                    uv_re.copy_to(re.data() + u, sve::experimental::vector_aligned);
                    uv_im.copy_to(im.data() + u, sve::experimental::vector_aligned);

                    uv_re = u_re - v_re;
                    uv_im = u_im - v_im;
                    uv_re.copy_to(re.data() + v, sve::experimental::vector_aligned);
                    uv_im.copy_to(im.data() + v, sve::experimental::vector_aligned);

                    complex_mul(w_re_init, w_im_init, 
                            simd_t(w_re.last()), simd_t(w_im.last()), w_re, w_im);
                    complex_mul(w_re, w_im, 
                            wlen_re, wlen_im, w_re, w_im);
                }
            }  
        }
    }

    if (invert) {
        if (n < (int)simd_t::size())
        {
            for (T & x : re)
                x /= n;
            for (T & x : im)
                x /= n;
        }
        else
        {
            for (int i = 0; i < n; i += simd_t::size())
            {
                simd_t v(re.data() + i, sve::experimental::vector_aligned);
                v /= n;
                v.copy_to(re.data() + i, sve::experimental::vector_aligned);
            }

            for (int i = 0; i < n; i += simd_t::size())
            {
                simd_t v(im.data() + i, sve::experimental::vector_aligned);
                v /= n;
                v.copy_to(im.data() + i, sve::experimental::vector_aligned);
            }
        }
    }
}

template <typename T>
double test_fft(T freq)
{
    using cd = std::complex<T>;
    auto x = gen_signal(freq);
    auto x_re = x;
    auto x_re_simd = x;
    std::vector<T> x_im(x_re.size(), 0);
    std::vector<T> x_im_simd(x_re_simd.size(), 0);
    auto t1 = std::chrono::high_resolution_clock::now();
        fft(x_re, x_im, false);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = t2 - t1;

    t1 = std::chrono::high_resolution_clock::now();
    fft_simd(x_re_simd, x_im_simd, false);
    t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff2 = t2 - t1;
    return diff.count() / diff2.count();
}

template <typename T>
void test(int iterations)
{
    std::fstream fout("fft.txt");
    using simd_t = sve::experimental::simd<T>;
    fout << "type\t\tsimd_len\t\tfrequency\t\tn_samples\t\tspeed_up\n";
    for (int i = 5; i <= 20; i++)
    {
        double avg = 0.0;
        for (int iter = 0; iter < iterations; ++iter)
        {
            avg += test_fft<T>(T(std::pow(2, i)));
        }
        avg /= iterations;
        fout << typeid(T).name() << "\t\t"
                << simd_t::size() << "\t\t"
                << std::pow(2, i) << "\t\t"
                << std::pow(2, i) * 16 << "\t\t"
                << avg << '\n';
    }
}

int main()
{
    test<float>(10);
    test<double>(10);
}