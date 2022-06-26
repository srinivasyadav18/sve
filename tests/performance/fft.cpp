#include <sve/sve.hpp>
#include <iostream>
#include <vector>
#include <numeric>

#include <complex>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <cassert>

using namespace std;
using namespace sve::experimental;
using cd = complex<double>;
const double PI = acos(-1);

template <typename T>
std::pair<T, T> complex_mul(T a, T b, T c, T d)
{
    return {a*c - b*d, a*d + b*c};
}

void fft(vector<double>& re, vector<double>& im, bool invert) {
    int n = re.size();

    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;

        if (i < j)
        {
            swap(re[i], re[j]);
            swap(im[i], im[j]);
        }
    }

    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * PI / len * (invert ? -1 : 1);
        // std::cout << "fft cos : " << cos(ang) << '\t' << "sin : " << sin(ang) << "\n";

        // cd wlen(cos(ang), sin(ang));
        double wlen_re = cos(ang);
        double wlen_im = sin(ang);
        for (int i = 0; i < n; i += len) {
            // cd w(1);

            double w_re = 1;
            double w_im = 0;

            for (int j = 0; j < len / 2; j++) {
                // cd u = a[i+j], v = a[i+j+len/2] * w;
                int u = i+j;
                int v = i+j+len/2;

                double u_re = re[u];
                double u_im = im[u];
                double v_re = re[v];
                double v_im = im[v];
                auto tmp = complex_mul(v_re, v_im, w_re, w_im);
                v_re = tmp.first;
                v_im = tmp.second;

                // a[i+j] = u + v;
                re[u] = u_re + v_re;
                im[u] = u_im + v_im;

                // a[i+j+len/2] = u - v;
                re[v] = u_re - v_re;
                im[v] = u_im - v_im;

                // std::cout << "[Debug fft bef : " 
                //     << "len = " << len 
                //     << " i = " << i 
                //     << " j = " << j
                //     << " w_re = " << w_re 
                //     << " w_im = " << w_im 
                //     << " ]\n";
                // w *= wlen;
                tmp = complex_mul(w_re, w_im, wlen_re, wlen_im);
                w_re = tmp.first;
                w_im = tmp.second;
                // std::cout << "[Debug fft : " 
                //         << "len = " << len 
                //         << " i = " << i 
                //         << " j = " << j
                //         << " w_re = " << w_re 
                //         << " w_im = " << w_im 
                //         << " ]\n";
            }
        }
    }

    if (invert) {
        for (double & x : re)
            x /= n;
        for (double & x : im)
            x /= n;
    }
}

void fft_simd(vector<double>& re, vector<double>& im, bool invert) {
    int n = re.size();

    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;

        if (i < j)
        {
            swap(re[i], re[j]);
            swap(im[i], im[j]);
        }
    }

    for (int len = 2; len <= n; len <<= 1) {
        if (len < 2 * (int)simd<double>::size())
        {
            // std::cout << "into serial code len : " << len << '\n';
            double ang = 2 * PI / len * (invert ? -1 : 1);
            // std::cout << "fft seq cos : " << cos(ang) << '\t' << "sin : " << sin(ang) << "\n";

            // cd wlen(cos(ang), sin(ang));
            double wlen_re = cos(ang);
            double wlen_im = sin(ang);
            for (int i = 0; i < n; i += len) {
                // cd w(1);

                double w_re = 1;
                double w_im = 0;

                for (int j = 0; j < len / 2; j++) {
                    // cd u = a[i+j], v = a[i+j+len/2] * w;
                    int u = i+j;
                    int v = i+j+len/2;

                    double u_re = re[u];
                    double u_im = im[u];
                    double v_re = re[v];
                    double v_im = im[v];
                    auto tmp = complex_mul(v_re, v_im, w_re, w_im);
                    v_re = tmp.first;
                    v_im = tmp.second;

                    // a[i+j] = u + v;
                    re[u] = u_re + v_re;
                    im[u] = u_im + v_im;

                    // a[i+j+len/2] = u - v;
                    re[v] = u_re - v_re;
                    im[v] = u_im - v_im;

                    // std::cout << "[Debug fft_simd seq bef : " 
                    //     << "len = " << len 
                    //     << " i = " << i 
                    //     << " j = " << j
                    //     << " w_re = " << w_re 
                    //     << " w_im = " << w_im 
                    //     << " ]\n";
                    // w *= wlen;
                    tmp = complex_mul(w_re, w_im, wlen_re, wlen_im);
                    w_re = tmp.first;
                    w_im = tmp.second;
                    // std::cout << "[Debug fft_simd seq : " 
                    //     << "len = " << len 
                    //     << " i = " << i 
                    //     << " j = " << j
                    //     << " w_re = " << w_re 
                    //     << " w_im = " << w_im 
                    //     << " ]\n";
                }
            }
        }
        else
        {
            // std::cout << "into simd code len : " << len << '\n';

            double ang = 2 * PI / len * (invert ? -1 : 1);
            // cd wlen(cos(ang), sin(ang));
            double cos_ang = cos(ang), sin_ang = sin(ang);
            // std::cout << "fft simd cos : " << cos_ang << '\t' << "sin : " << sin_ang << "\n";
            simd<double> wlen_re(cos_ang);
            simd<double> wlen_im(sin_ang);
            for (int i = 0; i < n; i += len) 
            {
                // cd w(1);

                // double w_re = 1;
                // double w_im = 0;
                vector<double> w_re_arr(simd<double>::size(), 0), w_im_arr(simd<double>::size(), 0);
                w_re_arr[0] = 1;
                w_im_arr[0] = 0;
                for (int k = 1; k < w_im_arr.size(); k++)
                {
                    auto tmp = complex_mul(w_re_arr[k-1], w_im_arr[k-1],cos_ang, sin_ang);
                    w_re_arr[k] = tmp.first;
                    w_im_arr[k] = tmp.second;
                }
                // std::cout << "[Debug start warr:";
                // for (int alpha = 0; alpha < w_im_arr.size(); alpha++)
                // {
                //     std::cout << w_re_arr[alpha] << " " << w_im_arr[alpha] << '\n';
                // }
                // std::cout << "Debug end warr]\n";

                simd<double> w_re(w_re_arr.data(), vector_aligned);
                simd<double> w_im(w_im_arr.data(), vector_aligned);
                auto w_re_init = w_re;
                auto w_im_init = w_im;
                for (int j = 0; j < len / 2; j += simd<double>::size()) 
                {
                    // cd u = a[i+j], v = a[i+j+len/2] * w;
                    int u = i+j;
                    int v = i+j+len/2;

                    simd<double> u_re(re.data() + u, vector_aligned);
                    simd<double> u_im(im.data() + u, vector_aligned);
                    simd<double> v_re(re.data() + v, vector_aligned);
                    simd<double> v_im(im.data() + v, vector_aligned);

                    auto tmp = complex_mul(v_re, v_im, w_re, w_im);
                    v_re = tmp.first;
                    v_im = tmp.second;

                    // a[i+j] = u + v;
                    auto uv_re = u_re + v_re;
                    auto uv_im = u_im + v_im;
                    uv_re.copy_to(re.data() + u, vector_aligned);
                    uv_im.copy_to(im.data() + u, vector_aligned);

                    // a[i+j+len/2] = u - v;
                    uv_re = u_re - v_re;
                    uv_im = u_im - v_im;
                    uv_re.copy_to(re.data() + v, vector_aligned);
                    uv_im.copy_to(im.data() + v, vector_aligned);

                    // std::cout << "[Debug fft_simd vec bef : " 
                    //     << "len = " << len 
                    //     << " i = " << i 
                    //     << " j = " << j
                    //     << " w_re = " << w_re 
                    //     << " w_im = " << w_im 
                    //     << " ]\n";
                    // w *= wlen;
                    // std::cout << "w_re last = " << w_re.last() << "\t" 
                    //         << "w_im last = " << w_im.last() << '\n';
                    tmp = complex_mul(w_re_init, w_im_init, 
                            simd<double>(w_re.last()), simd<double>(w_im.last()));
                    // std::cout << "w_re last = " << w_re.last() << "\t" 
                    //         << "w_im last = " << w_im.last() << '\n';
                    w_re = tmp.first;
                    w_im = tmp.second;
                    // std::cout << "[Debug fft_simd vec : " 
                    //     << "len = " << len 
                    //     << " i = " << i 
                    //     << " j = " << j
                    //     << " w_re = " << w_re 
                    //     << " w_im = " << w_im 
                    //     << " ]\n";
                    // std::cout << "[Debug fft_simd vec : "
                    //     << "wlen_re = " << wlen_re
                    //     << "wlen_im = " << wlen_im
                    //     << "]\n";
                    tmp = complex_mul(w_re, w_im, 
                            wlen_re, wlen_im);
                    w_re = tmp.first;
                    w_im = tmp.second;
                    // std::cout << "[Debug fft_simd vec : " 
                    //     << "len = " << len 
                    //     << " i = " << i 
                    //     << " j = " << j
                    //     << " w_re = " << w_re 
                    //     << " w_im = " << w_im 
                    //     << " ]\n";
                }
            }  
        }
    }

    if (invert) {
        for (double & x : re)
            x /= n;
        for (double & x : im)
            x /= n;
    }
}

vector<double> gen_signal(double sr)
{
    double ts = 1.0/sr;
    vector<double> ret;
    for (double i = 0; i < 1; i += ts)
    {
        ret.push_back(3.0 * sin(2.0 * PI * 2.0 * i));
    }
    return ret;
}

// void file_write(std::string file_name, )
void test_fft(double sr)
{
    auto x_re = gen_signal(sr);
    auto x_re_simd = gen_signal(sr);
    vector<double> x_im(x_re.size(), 0);
    vector<double> x_im_simd(x_re_simd.size(), 0);
    std::cout << "signal equal : " << ((x_re == x_re_simd) && (x_im == x_im_simd)) << '\n';

    // std::cout << "\n Input signal \n\n";
    // for (int i = 0; i < x_re.size(); i++)
    // {
    //     std::cout << x_re[i] << "," << x_im[i] << '\n';
    //     std::cout << x_re_simd[i] << "," << x_im_simd[i] << '\n';
    // }

    // std::cout << "\n FFT \n\n";
    fft(x_re, x_im, false);
    fft_simd(x_re_simd, x_im_simd, false);
    std::cout << "fft equal : " << ((x_re == x_re_simd) && (x_im == x_im_simd)) << '\n';
    for (int i = 0; i < x_re.size(); i++)
    {
        std::cout << x_re[i] << "," << x_im[i] << '\n';
        std::cout << x_re_simd[i] << "," << x_im_simd[i] << '\n';
    }

    std::cout << "\n Abs FFT \n\n";
    for (int i = 0; i < x_re.size(); i++)
    {
        std::cout << abs(cd(x_re[i], x_im[i])) << '\n';
        std::cout << abs(cd(x_re_simd[i], x_im_simd[i])) << '\n';
    }

    std::cout << "\n Inverse FFT \n\n";
    fft(x_re, x_im, true);
    fft_simd(x_re_simd, x_im_simd, true);
    for (int i = 0; i < x_re.size(); i++)
    {
        std::cout << x_re[i] << "," << x_im[i] << '\n';
        std::cout << x_re_simd[i] << "," << x_im_simd[i] << '\n';
    }

    std::cout << "\n Inverse Abs FFT \n\n";
    for (int i = 0; i < x_re.size(); i++)
    {
        std::cout << abs(cd(x_re[i], x_im[i])) << '\n';
        std::cout << abs(cd(x_re_simd[i], x_im_simd[i])) << '\n';
    }
}

int main()
{
    std::cout << "simd len : " << simd<double>::size() << '\n';
    test_fft(512);
}