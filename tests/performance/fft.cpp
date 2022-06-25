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

using cd = complex<double>;
const double PI = acos(-1);

void fft(vector<cd> & a, bool invert) {
    int n = a.size();

    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;

        if (i < j)
            swap(a[i], a[j]);
    }

    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * PI / len * (invert ? -1 : 1);
        cd wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            cd w(1);
            for (int j = 0; j < len / 2; j++) {
                cd u = a[i+j], v = a[i+j+len/2] * w;
                a[i+j] = u + v;
                a[i+j+len/2] = u - v;
                w *= wlen;
            }
        }
    }

    if (invert) {
        for (cd & x : a)
            x /= n;
    }
}

std::pair<double, double> complex_add(double a, double b, double c, double d)
{
    return {a+c, b+d};
}

std::pair<double, double> complex_sub(double a, double b, double c, double d)
{
    return {a-c, b-d};
}

std::pair<double, double> complex_mul(double a, double b, double c, double d)
{
    return {a*c - b*d, a*d + b*c};
}

template <typename T, typename U>
void my_assert(T t, U u)
{
    if (t != u) {
        std::cout << "Expected : " << t << '\n';
        std::cout << "Got : " << u << '\n';
    }
    assert(t == u);
}

void fft2(vector<double>& re, vector<double>& im, bool invert) {
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

                // w *= wlen;
                tmp = complex_mul(w_re, w_im, wlen_re, wlen_im);
                w_re = tmp.first;
                w_im = tmp.second;
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

vector<cd> gen_signal(double sr)
{
    double ts = 1.0/sr;
    vector<cd> ret;
    for (double i = 0; i < 1; i += ts)
    {
        ret.push_back(cd(3.0 * sin(2.0 * PI * 2.0 * i)));
    }
    
    return ret;
}

vector<double> gen_signal2(double sr)
{
    double ts = 1.0/sr;
    vector<double> ret;
    for (double i = 0; i < 1; i += ts)
    {
        ret.push_back(3.0 * sin(2.0 * PI * 2.0 * i));
    }
    return ret;
}

void do_fft(double sr)
{
    auto x = gen_signal(sr);
    for (auto i : x) {
        std::cout << i << '\n';
    }
    std::cout << "\n FFT \n\n";
    fft(x, false);
    for (auto i : x) {
        std::cout << i << '\n';
    }

    std::cout << "\n Abs FFT \n\n";
    for (auto i : x) {
        std::cout << abs(i) << '\n';
    }

    std::cout << "\n Inverse FFT \n\n";
    fft(x, true);
    for (auto i : x) {
        std::cout << i << '\n';
    }

    std::cout << "\n Abs FFT \n\n";
    for (auto i : x) {
        std::cout << abs(i) << '\n';
    }
}
void do_fft2(double sr)
{
    auto x_re = gen_signal2(sr);
    vector<double> x_im(x_re.size(), 0);

    for (int i = 0; i < x_re.size(); i++)
    {
        std::cout << x_re[i] << " " << x_im[i] << '\n';
    }

    std::cout << "\n FFT \n\n";
    fft2(x_re, x_im, false);
    for (int i = 0; i < x_re.size(); i++)
    {
        std::cout << x_re[i] << " " << x_im[i] << '\n';
    }

    std::cout << "\n Abs FFT \n\n";
    for (int i = 0; i < x_re.size(); i++)
    {
        std::cout << abs(cd(x_re[i], x_im[i])) << '\n';
    }

    std::cout << "\n Inverse FFT \n\n";
    fft2(x_re, x_im, true);
    for (int i = 0; i < x_re.size(); i++)
    {
        std::cout << x_re[i] << " " << x_im[i] << '\n';
    }

    std::cout << "\n Abs FFT \n\n";
    for (int i = 0; i < x_re.size(); i++)
    {
        std::cout << abs(cd(x_re[i], x_im[i])) << '\n';
    }
}
int main()
{
    do_fft(16);
    std::cout << "\n\n-----------------------\n\n\n";
    do_fft2(16);
}