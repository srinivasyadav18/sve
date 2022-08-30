#pragma once

#ifdef __ARM_FEATURE_SVE

#include <arm_sve.h>
#include <cstddef>
#include <functional>
#include <iostream>
#include <type_traits>
#include <unistd.h>

#if !defined(SVE_LEN)
#define SVE_LEN 128
#endif

namespace sve_impl {
    template <typename T>
    struct simd_impl
    {
    };

    static constexpr int max_vector_pack_size = SVE_LEN / 8;
    typedef svbool_t Predicate __attribute__((arm_sve_vector_bits(SVE_LEN)));

    // ----------------------------------------------------------------------
    template <int T>
    struct simd_impl_
    {
    };

    template <>
    struct simd_impl_<1>
    {
        inline static auto all_true()
        {
            return svptrue_b8();
        }
        inline static auto first_true()
        {
            return svptrue_pat_b8(SV_VL1);
        }
        inline static auto next_true(auto curr_true)
        {
            return svpnext_b8(svptrue_b8(), curr_true);
        }
        inline static auto count(auto pred)
        {
            return svcntp_b8(svptrue_b8(), pred);
        }
    };

    template <>
    struct simd_impl_<2>
    {
        inline static auto all_true()
        {
            return svptrue_b16();
        }
        inline static auto first_true()
        {
            return svptrue_pat_b16(SV_VL1);
        }
        inline static auto next_true(auto curr_true)
        {
            return svpnext_b16(svptrue_b16(), curr_true);
        }
        inline static auto count(auto pred)
        {
            return svcntp_b16(svptrue_b16(), pred);
        }
    };

    template <>
    struct simd_impl_<4>
    {
        inline static auto all_true()
        {
            return svptrue_b32();
        }
        inline static auto first_true()
        {
            return svptrue_pat_b32(SV_VL1);
        }
        inline static auto next_true(auto curr_true)
        {
            return svpnext_b32(svptrue_b32(), curr_true);
        }
        inline static auto count(auto pred)
        {
            return svcntp_b32(svptrue_b32(), pred);
        }
    };

    template <>
    struct simd_impl_<8>
    {
        inline static auto all_true()
        {
            return svptrue_b64();
        }
        inline static auto first_true()
        {
            return svptrue_pat_b64(SV_VL1);
        }
        inline static auto next_true(auto curr_true)
        {
            return svpnext_b64(svptrue_b64(), curr_true);
        }
        inline static auto count(auto pred)
        {
            return svcntp_b64(svptrue_b64(), pred);
        }
    };

    // ----------------------------------------------------------------------
    template <>
    struct simd_impl<int8_t>
    {
        using T = int8_t;
        typedef svint8_t Vector __attribute__((arm_sve_vector_bits(SVE_LEN)));
        static constexpr std::size_t size = max_vector_pack_size / sizeof(T);

        inline static Vector set(Vector vec, Predicate index, T val)
        {
            return svdup_s8_m(vec, index, val);
        }
        inline static Vector fill(T val)
        {
            return svdup_s8(val);
        }
        inline static Vector index_series(T base, T step)
        {
            return svindex_s8(base, step);
        }
        inline static const Vector index0123 = svindex_s8(T(0), T(1));
    };

    template <>
    struct simd_impl<uint8_t>
    {
        using T = uint8_t;
        typedef svuint8_t Vector __attribute__((arm_sve_vector_bits(SVE_LEN)));
        static constexpr std::size_t size = max_vector_pack_size / sizeof(T);

        inline static Vector set(Vector vec, Predicate index, T val)
        {
            return svdup_u8_m(vec, index, val);
        }
        inline static Vector fill(T val)
        {
            return svdup_u8(val);
        }
        inline static Vector index_series(T base, T step)
        {
            return svindex_u8(base, step);
        }
        inline static const Vector index0123 = svindex_u8(T(0), T(1));
    };

    template <>
    struct simd_impl<int16_t>
    {
        using T = int16_t;
        typedef svint16_t Vector __attribute__((arm_sve_vector_bits(SVE_LEN)));
        static constexpr std::size_t size = max_vector_pack_size / sizeof(T);

        inline static Vector set(Vector vec, Predicate index, T val)
        {
            return svdup_s16_m(vec, index, val);
        }
        inline static Vector fill(T val)
        {
            return svdup_s16(val);
        }
        inline static Vector index_series(T base, T step)
        {
            return svindex_s16(base, step);
        }
        inline static const Vector index0123 = svindex_s16(T(0), T(1));
    };

    template <>
    struct simd_impl<uint16_t>
    {
        using T = uint16_t;
        typedef svuint16_t Vector __attribute__((arm_sve_vector_bits(SVE_LEN)));
        static constexpr std::size_t size = max_vector_pack_size / sizeof(T);

        inline static Vector set(Vector vec, Predicate index, T val)
        {
            return svdup_u16_m(vec, index, val);
        }
        inline static Vector fill(T val)
        {
            return svdup_u16(val);
        }
        inline static Vector index_series(T base, T step)
        {
            return svindex_u16(base, step);
        }
        inline static const Vector index0123 = svindex_u16(T(0), T(1));
    };

    template <>
    struct simd_impl<int32_t>
    {
        using T = int32_t;
        typedef svint32_t Vector __attribute__((arm_sve_vector_bits(SVE_LEN)));
        static constexpr std::size_t size = max_vector_pack_size / sizeof(T);

        inline static Vector set(Vector vec, Predicate index, T val)
        {
            return svdup_s32_m(vec, index, val);
        }
        inline static Vector fill(T val)
        {
            return svdup_s32(val);
        }
        inline static Vector index_series(T base, T step)
        {
            return svindex_s32(base, step);
        }
        inline static const Vector index0123 = svindex_s32(T(0), T(1));
    };

    template <>
    struct simd_impl<uint32_t>
    {
        using T = uint32_t;
        typedef svuint32_t Vector __attribute__((arm_sve_vector_bits(SVE_LEN)));
        static constexpr std::size_t size = max_vector_pack_size / sizeof(T);

        inline static Vector set(Vector vec, Predicate index, T val)
        {
            return svdup_u32_m(vec, index, val);
        }
        inline static Vector fill(T val)
        {
            return svdup_u32(val);
        }
        inline static Vector index_series(T base, T step)
        {
            return svindex_u32(base, step);
        }
        inline static const Vector index0123 = svindex_u32(T(0), T(1));
    };

    template <>
    struct simd_impl<int64_t>
    {
        using T = int64_t;
        typedef svint64_t Vector __attribute__((arm_sve_vector_bits(SVE_LEN)));
        static constexpr std::size_t size = max_vector_pack_size / sizeof(T);

        inline static Vector set(Vector vec, Predicate index, T val)
        {
            return svdup_s64_m(vec, index, val);
        }
        inline static Vector fill(T val)
        {
            return svdup_s64(val);
        }
        inline static Vector index_series(T base, T step)
        {
            return svindex_s64(base, step);
        }
        inline static const Vector index0123 = svindex_s64(T(0), T(1));
    };

    template <>
    struct simd_impl<uint64_t>
    {
        using T = uint64_t;
        typedef svuint64_t Vector __attribute__((arm_sve_vector_bits(SVE_LEN)));
        static constexpr std::size_t size = max_vector_pack_size / sizeof(T);

        inline static Vector set(Vector vec, Predicate index, T val)
        {
            return svdup_u64_m(vec, index, val);
        }
        inline static Vector fill(T val)
        {
            return svdup_u64(val);
        }
        inline static Vector index_series(T base, T step)
        {
            return svindex_u64(base, step);
        }
        inline static const Vector index0123 = svindex_u64(T(0), T(1));
    };

    // ----------------------------------------------------------------------
    template <>
    struct simd_impl<float16_t>
    {
        typedef svfloat16_t Vector
            __attribute__((arm_sve_vector_bits(SVE_LEN)));
        static constexpr std::size_t size =
            max_vector_pack_size / sizeof(float16_t);

        inline static Vector set(Vector vec, Predicate index, float16_t val)
        {
            return svdup_f16_m(vec, index, val);
        }
        inline static Vector fill(float16_t val)
        {
            return svdup_f16(val);
        }
        inline static const float16_t iota_array[32] = {0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31};
        inline static const Vector index0123 = svld1(
            sve_impl::simd_impl_<sizeof(float16_t)>::all_true(), iota_array);
    };

    template <>
    struct simd_impl<float>
    {
        typedef svfloat32_t Vector
            __attribute__((arm_sve_vector_bits(SVE_LEN)));
        static constexpr std::size_t size =
            max_vector_pack_size / sizeof(float);

        inline static Vector set(Vector vec, Predicate index, float val)
        {
            return svdup_f32_m(vec, index, val);
        }
        inline static Vector fill(float val)
        {
            return svdup_f32(val);
        }
        inline static const float iota_array[16] = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        inline static const Vector index0123 =
            svld1(sve_impl::simd_impl_<sizeof(float)>::all_true(), iota_array);
    };

    template <>
    struct simd_impl<double>
    {
        typedef svfloat64_t Vector
            __attribute__((arm_sve_vector_bits(SVE_LEN)));
        static constexpr std::size_t size =
            max_vector_pack_size / sizeof(double);

        inline static Vector set(Vector vec, Predicate index, double val)
        {
            return svdup_f64_m(vec, index, val);
        }
        inline static Vector fill(double val)
        {
            return svdup_f64(val);
        }
        inline static const double iota_array[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        inline static const Vector index0123 =
            svld1(sve_impl::simd_impl_<sizeof(double)>::all_true(), iota_array);
    };
}    // namespace sve_impl

namespace sve::experimental { inline namespace parallelism_v2 {
    namespace simd_abi {
        struct scalar
        {
        };
        struct sve_abi
        {
        };
        // template <int N> struct fixed_size {};

        template <typename T>
        inline constexpr int max_fixed_size = sve_impl::max_vector_pack_size;

        template <typename T>
        using compatible = sve_abi;

        template <typename T>
        using native = sve_abi;

        template <typename T, size_t N>
        using fixed_size = sve_abi;

        // template <class T, size_t N>
        // struct deduce { using type = sve_abi; };

        // template <class T, size_t N> using deduce_t = typename deduce<T, N>::type;
    }    // namespace simd_abi

    struct element_aligned_tag
    {
    };
    struct vector_aligned_tag
    {
    };
    template <size_t>
    struct overaligned_tag
    {
    };
    inline constexpr element_aligned_tag element_aligned{};
    inline constexpr vector_aligned_tag vector_aligned{};
    template <size_t N>
    inline constexpr overaligned_tag<N> overaligned{};

    // ----------------------------------------------------------------------
    // traits [simd.traits]
    // ----------------------------------------------------------------------
    template <class T>
    struct is_abi_tag : std::is_same<T, simd_abi::sve_abi>
    {
    };
    template <class T>
    inline constexpr bool is_abi_tag_v = is_abi_tag<T>::value;

    template <class T>
    struct is_simd;
    template <class T>
    inline constexpr bool is_simd_v = is_simd<T>::value;

    template <class T>
    struct is_simd_mask;
    template <class T>
    inline constexpr bool is_simd_mask_v = is_simd_mask<T>::value;

    template <class T>
    struct is_simd_flag_type : std::false_type
    {
    };
    template <>
    struct is_simd_flag_type<element_aligned_tag> : std::true_type
    {
    };
    template <>
    struct is_simd_flag_type<vector_aligned_tag> : std::true_type
    {
    };
    template <class T>
    inline constexpr bool is_simd_flag_type_v = is_simd_flag_type<T>::value;

    template <class T, class Abi = simd_abi::compatible<T>>
    struct simd_size
    {
        static inline constexpr size_t value = sve_impl::simd_impl<T>::size;
    };
    template <class T, class Abi = simd_abi::compatible<T>>
    inline constexpr size_t simd_size_v = simd_size<T, Abi>::value;

    template <class T, class U = typename T::value_type>
    struct memory_alignment
    {
        static inline constexpr size_t value = sve_impl::max_vector_pack_size;
    };
    template <class T, class U = typename T::value_type>
    inline constexpr size_t memory_alignment_v = memory_alignment<T, U>::value;

    // ----------------------------------------------------------------------
    // class template simd [simd.class]
    // ----------------------------------------------------------------------
    template <class T, class Abi = simd_abi::compatible<T>>
    class simd;
    template <class T>
    using native_simd = simd<T, simd_abi::native<T>>;
    template <class T, size_t N>
    using fixed_size_simd = simd<T, simd_abi::fixed_size<T, N>>;

    // ----------------------------------------------------------------------
    // class template simd_mask [simd.mask.class]
    // ----------------------------------------------------------------------
    template <class T, class Abi = simd_abi::compatible<T>>
    class simd_mask;
    template <class T>
    using native_simd_mask = simd_mask<T, simd_abi::native<T>>;

    template <class T>
    struct is_simd : std::false_type
    {
    };

    template <typename T, typename Abi>
    struct is_simd<simd<T, Abi>> : std::true_type
    {
    };

    template <class T>
    struct is_simd_mask : std::false_type
    {
    };

    template <typename T, typename Abi>
    struct is_simd_mask<simd_mask<T, Abi>> : std::true_type
    {
    };

    // class template simd
    template <typename T, typename Abi>
    class simd
    {
    private:
        using Vector = typename sve_impl::simd_impl<T>::Vector;
        using Predicate = typename sve_impl::Predicate;
        Vector vec;
        static inline constexpr int T_size = sizeof(T);
        static inline const Predicate all_true =
            sve_impl::simd_impl_<T_size>::all_true();

    public:
        using value_type = T;
        using abi_type = Abi;
        using mask_type = simd_mask<T, Abi>;

        static inline const Vector index0123 =
            sve_impl::simd_impl<T>::index0123;

        static inline constexpr std::size_t size()
        {
            return sve_impl::simd_impl<T>::size;
        }

        // ----------------------------------------------------------------------
        //  constructors
        // ----------------------------------------------------------------------
        inline simd(const simd&) = default;
        inline simd(simd&&) noexcept = default;
        inline simd& operator=(const simd&) = default;
        inline simd& operator=(simd&&) noexcept = default;

        template <typename U, typename Flag>
        inline simd(U* ptr, Flag)
        {
            static_assert(std::is_same_v<std::remove_cvref_t<U>, T>,
                "pointer should be same type as value_type");
            static_assert(is_simd_flag_type_v<Flag>,
                "use element_aligned or vector_aligned tag");

            vec = svld1(all_true, ptr);
        }

        inline simd(T val = {})
        {
            vec = sve_impl::simd_impl<T>::fill(val);
        }

        inline simd(Vector v)
        {
            vec = v;
        }

        // ----------------------------------------------------------------------
        //  load and store
        // ----------------------------------------------------------------------
        template <typename U, typename Flag>
        inline void copy_from(const U* ptr, Flag)
        {
            static_assert(std::is_same_v<std::remove_cvref_t<U>, T>,
                "pointer should be same type as value_type");
            static_assert(is_simd_flag_type_v<Flag>,
                "use element_aligned or vector_aligned tag");
            vec = svld1(all_true, ptr);
        }

        template <typename U, typename Flag>
        inline void copy_to(U* ptr, Flag) const
        {
            static_assert(std::is_same_v<std::remove_cvref_t<U>, T>,
                "pointer should be same type as value_type");
            static_assert(is_simd_flag_type_v<Flag>,
                "use element_aligned or vector_aligned tag");
            svst1(all_true, ptr, vec);
        }

        // ----------------------------------------------------------------------
        //  get and set
        // ----------------------------------------------------------------------
        T get(int idx) const
        {
            if (idx < 0 || idx > (int) size())
                return -1;
            return svlasta(svcmplt(all_true, index0123, T(idx)), vec);
        }

        T operator[](int idx) const
        {
            if (idx < 0 || idx > (int) size())
                return -1;
            return svlasta(svcmplt(all_true, index0123, T(idx)), vec);
        }

        void set(int idx, T val)
        {
            if (idx < 0 || idx > (int) size())
                return;
            vec = sve_impl::simd_impl<value_type>::set(
                vec, svcmpeq(all_true, index0123, T(idx)), val);
        }

        // ----------------------------------------------------------------------
        // ostream overload
        // ----------------------------------------------------------------------
        friend std::ostream& operator<<(std::ostream& os, const simd& x)
        {
            using type_ = std::decay_t<decltype(x)>;
            using value_type_ = typename type_::value_type;
            using printable_type =
                std::conditional_t<std::is_integral_v<value_type_>,
                    std::conditional_t<std::is_unsigned_v<value_type_>,
                        uint32_t, int32_t>,
                    value_type_>;

            os << "( ";
            for (int i = 0; i < (int) x.size(); i++)
            {
                os << printable_type(x[i]) << ' ';
            }
            os << ")";
            return os;
        }

        // ----------------------------------------------------------------------
        //  Reduction
        // ----------------------------------------------------------------------
        inline auto addv() const
        {
            return svaddv(all_true, vec);
        }

        // ----------------------------------------------------------------------
        //  First and last elements
        // ----------------------------------------------------------------------
        inline auto first() const
        {
            return svlasta(svpfalse(), vec);
        }

        inline auto last() const
        {
            return svlastb(svpfalse(), vec);
        }

        // ----------------------------------------------------------------------
        //  unary operators [simd.unary]
        // ----------------------------------------------------------------------
        inline simd& operator++()
        {
            vec = svadd_x(all_true, vec, static_cast<T>(1));
            return *this;
        }

        inline auto operator++(int)
        {
            auto vec_copy = *this;
            vec = svadd_x(all_true, vec, static_cast<T>(1));
            return vec_copy;
        }

        inline simd& operator--()
        {
            vec = svsub_x(all_true, vec, static_cast<T>(1));
            return *this;
        }

        inline auto operator--(int)
        {
            auto vec_copy = *this;
            vec = svsub_x(all_true, vec, static_cast<T>(1));
            return vec_copy;
        }

        inline simd operator+() const
        {
            return *this;
        }

        inline simd operator-() const
        {
            auto vec_copy = *this;
            vec_copy.vec =
                svmul_x(vec_copy.all_true, vec_copy.vec, static_cast<T>(-1));
            return vec_copy;
        }

        // ----------------------------------------------------------------------
        // binary operators [simd.binary]
        // ----------------------------------------------------------------------
        inline friend simd operator+(const simd& x, const simd& y)
        {
            return svadd_x(x.all_true, x.vec, y.vec);
        }

        inline friend simd operator-(const simd& x, const simd& y)
        {
            return svsub_x(x.all_true, x.vec, y.vec);
        }

        inline friend simd operator*(const simd& x, const simd& y)
        {
            return svmul_x(x.all_true, x.vec, y.vec);
        }

        inline friend simd operator/(const simd& x, const simd& y)
        {
            return svdiv_x(x.all_true, x.vec, y.vec);
        }

        inline friend simd operator&(const simd& x, const simd& y)
        {
            static_assert(std::is_integral_v<T>,
                "operator& only works for integeral types");
            return svand_x(x.all_true, x.vec, y.vec);
        }

        inline friend simd operator|(const simd& x, const simd& y)
        {
            static_assert(std::is_integral_v<T>,
                "operator| only works for integeral types");
            return svorr_x(x.all_true, x.vec, y.vec);
        }

        inline friend simd operator^(const simd& x, const simd& y)
        {
            static_assert(std::is_integral_v<T>,
                "operator^ only works for integeral types");
            return sveor_x(x.all_true, x.vec, y.vec);
        }

        // friend simd operator<<(const simd& x, typename std::make_unsigned<T>::type
        // n)
        // {
        //     return left_shift(x.vec, n);
        // }

        // friend std::enable_if_t<std::is_unsigned_v<T>, simd>
        // friend simd operator>>(const simd& x, T n)
        // {
        //     return right_shift(x.vec, n);
        // }

        // ----------------------------------------------------------------------
        // compound assignment [simd.cassign]
        // ----------------------------------------------------------------------
        inline friend simd& operator+=(simd& x, const simd& y)
        {
            x.vec = svadd_x(x.all_true, x.vec, y.vec);
            return x;
        }

        inline friend simd& operator-=(simd& x, const simd& y)
        {
            x.vec = svsub_x(x.all_true, x.vec, y.vec);
            return x;
        }

        inline friend simd& operator*=(simd& x, const simd& y)
        {
            x.vec = svmul_x(x.all_true, x.vec, y.vec);
            return x;
        }

        inline friend simd& operator/=(simd& x, const simd& y)
        {
            x.vec = svdiv_x(x.all_true, x.vec, y.vec);
            return x;
        }

        inline friend simd operator&=(simd& x, const simd& y)
        {
            static_assert(std::is_integral_v<T>,
                "operator&= only works for integeral types");
            x.vec = svand_x(x.all_true, x.vec, y.vec);
            return x;
        }

        inline friend simd operator|=(simd& x, const simd& y)
        {
            static_assert(std::is_integral_v<T>,
                "operator|= only works for integeral types");
            x.vec = svorr_x(x.all_true, x.vec, y.vec);
            return x;
        }

        inline friend simd operator^=(simd& x, const simd& y)
        {
            static_assert(std::is_integral_v<T>,
                "operator^= only works for integeral types");
            x.vec = sveor_x(x.all_true, x.vec, y.vec);
            return x;
        }

        // friend simd operator<<=(simd& x, typename std::make_unsigned<T>::type n)
        // {
        //     x.vec = left_shift(x.vec, n);
        //     return x;
        // }

        // friend simd operator>>=(simd& x, T n)
        // {
        //     x.vec = right_shift(x.vec, n);
        //     return x;
        // }

        // ----------------------------------------------------------------------
        // compares [simd.comparison]
        // ----------------------------------------------------------------------
        inline friend mask_type operator==(const simd& x, const simd& y)
        {
            return svcmpeq(x.all_true, x.vec, y.vec);
        }

        inline friend mask_type operator!=(const simd& x, const simd& y)
        {
            return svcmpne(x.all_true, x.vec, y.vec);
        }

        inline friend mask_type operator>=(const simd& x, const simd& y)
        {
            return svcmpge(x.all_true, x.vec, y.vec);
        }

        inline friend mask_type operator<=(const simd& x, const simd& y)
        {
            return svcmple(x.all_true, x.vec, y.vec);
        }

        inline friend mask_type operator>(const simd& x, const simd& y)
        {
            return svcmpgt(x.all_true, x.vec, y.vec);
        }

        inline friend mask_type operator<(const simd& x, const simd& y)
        {
            return svcmplt(x.all_true, x.vec, y.vec);
        }

        // ----------------------------------------------------------------------
        // reduction algorithms
        // ----------------------------------------------------------------------
        inline T sum() const
        {
            return svaddv(all_true, vec);
        }

        inline T min() const
        {
            return svminv(all_true, vec);
        }

        inline T max() const
        {
            return svmaxv(all_true, vec);
        }

    private:
        template <typename T_, typename Abi_>
        inline friend simd<T_, Abi_> choose(const simd_mask<T_, Abi_>& msk,
            const simd<T_, Abi_>& t, const simd<T_, Abi_>& f);

        template <typename T_, typename Abi_>
        inline friend void mask_assign(const simd_mask<T_, Abi_>& msk,
            simd<T_, Abi_>& v, const simd<T_, Abi_>& val);

        template <typename T_, typename Abi_>
        inline friend simd<T_, Abi_> min(
            const simd<T_, Abi_>& x, const simd<T_, Abi_>& y);

        template <typename T_, typename Abi_>
        inline friend simd<T_, Abi_> max(
            const simd<T_, Abi_>& x, const simd<T_, Abi_>& y);

        template <typename t_, typename abi_>
        inline friend simd<t_, abi_> copysign(const simd<t_, abi_>& valSrc, const simd<t_, abi_>& signSrc);

        template <typename T_, typename Abi_>
        inline friend simd<T_, Abi_> sqrt(const simd<T_, Abi_>& x);

        template <typename T_, typename Abi_>
        inline friend simd<T_, Abi_> abs(const simd<T_, Abi_>& x);

        template <typename T_, typename Abi_>
        inline friend simd<T_, Abi_> fma(const simd<T_, Abi_>& a,
            const simd<T_, Abi_>& b, const simd<T_, Abi_>& z);

        template <typename T_, typename Abi_, typename Op>
        inline friend T_ reduce(const simd<T_, Abi_>& x, Op op);

        template <typename T_, typename Abi_, typename Op>
        inline friend simd<T_, Abi_> inclusive_scan(
            const simd<T_, Abi_>& x, Op op);

        template <typename T_, typename Abi_, typename Op>
        inline friend simd<T_, Abi_> exclusive_scan(
            const simd<T_, Abi_>& x, Op op, T_ init);

        template <typename T_, typename Abi_>
        inline friend simd<T_, Abi_> index_series(T_ base, T_ step);

        template <typename T_, typename Abi_>
        inline friend simd<T_, Abi_> compact(
            const simd_mask<T_, Abi_>& msk, const simd<T_, Abi_>& v);

        template <typename T_, typename Abi_>
        inline friend simd<T_, Abi_> splice(const simd_mask<T_, Abi_>& msk,
            const simd<T_, Abi_>& x, const simd<T_, Abi_>& y);

        template <typename T_, typename Abi_>
        inline friend simd<T_, Abi_> interleave_even(
            const simd<T_, Abi_>& x, const simd<T_, Abi_>& y);

        template <typename T_, typename Abi_>
        inline friend simd<T_, Abi_> interleave_odd(
            const simd<T_, Abi_>& x, const simd<T_, Abi_>& y);

        template <typename T_, typename Abi_>
        inline friend simd<T_, Abi_> select_even(
            const simd<T_, Abi_>& x, const simd<T_, Abi_>& y);

        template <typename T_, typename Abi_>
        inline friend simd<T_, Abi_> select_odd(
            const simd<T_, Abi_>& x, const simd<T_, Abi_>& y);

        template <typename T_, typename Abi_>
        inline friend simd<T_, Abi_> lower_half(
            const simd<T_, Abi_>& x, const simd<T_, Abi_>& y);

        template <typename T_, typename Abi_>
        inline friend simd<T_, Abi_> upper_half(
            const simd<T_, Abi_>& x, const simd<T_, Abi_>& y);

        template <typename T_, typename Abi_>
        inline friend simd<T_, Abi_> reverse(const simd<T_, Abi_>& x);
    };

    template <typename T, typename Abi>
    inline simd<T, Abi> min(const simd<T, Abi>& x, const simd<T, Abi>& y)
    {
        // return svsel(svcmplt(x.all_true, x.vec, y.vec), x.vec, y.vec);
        return svmin_z(x.all_true, x.vec, y.vec);
    }

    template <typename T, typename Abi>
    inline simd<T, Abi> max(const simd<T, Abi>& x, const simd<T, Abi>& y)
    {
        // return svsel(svcmpgt(x.all_true, x.vec, y.vec), x.vec, y.vec);
        return svmax_z(x.all_true, x.vec, y.vec);
    }

    template <typename T, typename Abi>
    inline std::pair<simd<T, Abi>, simd<T, Abi>> minmax(
        const simd<T, Abi>& x, const simd<T, Abi>& y)
    {
        return {min(x, y), max(x, y)};
    }

    template <typename T_, typename Abi_>
    inline simd<T_, Abi_> copysign(const simd<T_, Abi_>& valSrc, const simd<T_, Abi_>& signSrc) {
        static_assert(
            std::is_same_v<T_, float64_t>,
            "vectorized copysign is only implemented for f64 types thus far.");
        // obtain signbits by using a static -0.0 mask to find them in the signSrc via a bitwise AND
        const auto signmask = sve_impl::simd_impl<T_>::fill(-0.0);
        const auto signbits = svand_x(signSrc.all_true, svreinterpret_s64(signSrc.vec), svreinterpret_s64(signmask));
        // obtains the valuebits by using the inverted signmask via a bitwise ANDNOT
        const auto absbits = svbic_x(signSrc.all_true, svreinterpret_s64(valSrc.vec), svreinterpret_s64(signmask));
        // Results is just the two bitsequeences combined via a bitwise OR
        const auto result = svorr_x(signSrc.all_true, signbits, absbits);
        // Return bit sequence as a f64 type
        return svreinterpret_f64(result);
    }

    template <typename T_, typename Abi_>
    inline simd<T_, Abi_> sqrt(const simd<T_, Abi_>& x)
    {
        static_assert(
            std::is_floating_point_v<T_> || std::is_same_v<T_, float16_t>,
            "sqrt only works this floating point types");
        return svsqrt_x(x.all_true, x.vec);
    }

    template <typename T_, typename Abi_>
    inline simd<T_, Abi_> abs(const simd<T_, Abi_>& x)
    {
        return svabs_x(x.all_true, x.vec);
    }

    template <typename T_, typename Abi_>
    inline simd<T_, Abi_> fma(const simd<T_, Abi_>& a, const simd<T_, Abi_>& b,
        const simd<T_, Abi_>& z)
    {
        return svmad_m(a.all_true, a.vec, b.vec, z.vec);
    }

    template <typename T, typename Abi, typename Op = std::plus<>>
    inline T reduce(const simd<T, Abi>& x, Op op = {})
    {
        using simd_t = simd<T, Abi>;
        auto x_vec = x.vec;

        for (int i = 1; i < simd_t::size(); i *= 2)
        {
            x_vec =
                op(simd_t(svzip1(x_vec, x_vec)), simd_t(svzip2(x_vec, x_vec)))
                    .vec;
        }
        return simd_t(x_vec).first();
    }

    template <typename T_, typename Abi_, typename Op = std::plus<>>
    inline simd<T_, Abi_> inclusive_scan(const simd<T_, Abi_>& x, Op op = {})
    {
        using simd_t = simd<T_, Abi_>;
        auto x_vec = x.vec;
        auto iota_vec = sve_impl::simd_impl<T_>::index0123;
        auto local_all_true = x.all_true;
        if constexpr (simd_t::size() >= 2)
        {
            x_vec = svsplice(svcmplt(local_all_true, iota_vec, T_(1)), x_vec,
                op(simd_t(x_vec), simd_t(svext(x_vec, x_vec, 1))).vec);
        }
        if constexpr (simd_t::size() >= 4)
        {
            x_vec = svsplice(svcmplt(local_all_true, iota_vec, T_(2)), x_vec,
                op(simd_t(x_vec), simd_t(svext(x_vec, x_vec, 2))).vec);
        }
        if constexpr (simd_t::size() >= 8)
        {
            x_vec = svsplice(svcmplt(local_all_true, iota_vec, T_(4)), x_vec,
                op(simd_t(x_vec), simd_t(svext(x_vec, x_vec, 4))).vec);
        }
        if constexpr (simd_t::size() >= 16)
        {
            x_vec = svsplice(svcmplt(local_all_true, iota_vec, T_(8)), x_vec,
                op(simd_t(x_vec), simd_t(svext(x_vec, x_vec, 8))).vec);
        }
        if constexpr (simd_t::size() >= 32)
        {
            x_vec = svsplice(svcmplt(local_all_true, iota_vec, T_(16)), x_vec,
                op(simd_t(x_vec), simd_t(svext(x_vec, x_vec, 16))).vec);
        }
        if constexpr (simd_t::size() == 64)
        {
            x_vec = svsplice(svcmplt(local_all_true, iota_vec, T_(32)), x_vec,
                op(simd_t(x_vec), simd_t(svext(x_vec, x_vec, 32))).vec);
        }
        return x_vec;
    }

    template <typename T_, typename Abi_, typename Op = std::plus<>>
    inline simd<T_, Abi_> exclusive_scan(
        const simd<T_, Abi_>& x, Op op = {}, T_ init = {})
    {
        auto x_vec = x.vec;
        x_vec = svinsr(x.vec, init);
        return inclusive_scan(simd<T_, Abi_>(x_vec));
    }

    template <typename T_, typename Abi_ = simd_abi::sve_abi>
    inline simd<T_, Abi_> index_series(T_ base, T_ step)
    {
        if constexpr (std::is_floating_point_v<T_> ||
            std::is_same_v<T_, float16_t>)
        {
            return exclusive_scan(simd<T_, Abi_>(step)) + base;
        }
        else
        {
            return sve_impl::simd_impl<T_>::index_series(base, step);
        }
    }

    template <typename T_, typename Abi_>
    inline simd<T_, Abi_> interleave_even(
        const simd<T_, Abi_>& x, const simd<T_, Abi_>& y)
    {
        return svtrn1(x.vec, y.vec);
    }

    template <typename T_, typename Abi_>
    inline simd<T_, Abi_> interleave_odd(
        const simd<T_, Abi_>& x, const simd<T_, Abi_>& y)
    {
        return svtrn2(x.vec, y.vec);
    }

    template <typename T_, typename Abi_>
    inline simd<T_, Abi_> select_even(
        const simd<T_, Abi_>& x, const simd<T_, Abi_>& y)
    {
        return svuzp1(x.vec, y.vec);
    }

    template <typename T_, typename Abi_>
    inline simd<T_, Abi_> select_odd(
        const simd<T_, Abi_>& x, const simd<T_, Abi_>& y)
    {
        return svuzp2(x.vec, y.vec);
    }

    template <typename T_, typename Abi_>
    inline simd<T_, Abi_> lower_half(
        const simd<T_, Abi_>& x, const simd<T_, Abi_>& y)
    {
        return svzip1(x.vec, y.vec);
    }

    template <typename T_, typename Abi_>
    inline simd<T_, Abi_> upper_half(
        const simd<T_, Abi_>& x, const simd<T_, Abi_>& y)
    {
        return svzip2(x.vec, y.vec);
    }

    template <typename T_, typename Abi_>
    inline simd<T_, Abi_> reverse(const simd<T_, Abi_>& x)
    {
        return svrev(x.vec);
    }

    template <typename T, typename Abi>
    class simd_mask
    {
    private:
        using Predicate = typename sve_impl::Predicate;
        static inline constexpr int T_size = sizeof(T);
        Predicate pred;
        static inline const Predicate all_true =
            sve_impl::simd_impl_<T_size>::all_true();
        ;

    public:
        using value_type = bool;
        using simd_type = simd<T, Abi>;
        using abi_type = Abi;

        static inline const auto index0123 = simd_type::index0123;
        static inline constexpr std::size_t size()
        {
            return simd<T, Abi>::size();
        }

        // ----------------------------------------------------------------------
        //  constructors
        // ----------------------------------------------------------------------
        inline simd_mask(const simd_mask&) = default;
        inline simd_mask(simd_mask&&) = default;
        inline simd_mask& operator=(const simd_mask&) = default;
        inline simd_mask& operator=(simd_mask&&) = default;

        inline simd_mask(bool val = false)
        {
            if (val)
            {
                pred = sve_impl::simd_impl_<T_size>::all_true();
            }
            else
                pred = svpfalse();
        }

        inline simd_mask(Predicate p)
        {
            pred = p;
        }

        // ----------------------------------------------------------------------
        //  get and set
        // ----------------------------------------------------------------------
        bool get(int idx) const
        {
            if (idx < 0 || idx > (int) size())
                return -1;

            return sve_impl::simd_impl_<T_size>::count(
                svand_z(all_true, svcmpeq(all_true, index0123, T(idx)), pred));
        }

        bool operator[](int idx) const
        {
            return get(idx);
        }

        void set(int idx, bool val)
        {
            if (idx < 0 || idx > (int) size())
                return;

            auto index = svcmpeq(all_true, index0123, T(idx));
            if (val)
                pred = svorr_z(all_true, pred, index);
            else
                pred = svbic_z(all_true, pred, index);
        }

        // ----------------------------------------------------------------------
        // ostream overload
        // ----------------------------------------------------------------------
        friend std::ostream& operator<<(std::ostream& os, const simd_mask& x)
        {
            using type_ = std::decay_t<decltype(x)>;
            using simd_type_ = typename type_::simd_type;
            using value_type_ = typename simd_type::value_type;

            os << "( ";
            for (int i = 0; i < (int) x.size(); i++)
            {
                os << x[i] << ' ';
            }
            os << ")";
            return os;
        }

        // ----------------------------------------------------------------------
        //  unary operators
        // ----------------------------------------------------------------------
        inline simd_mask operator!() const noexcept
        {
            return svnot_z(all_true, pred);
        }

        // ----------------------------------------------------------------------
        //  binary operators
        // ----------------------------------------------------------------------
        inline friend simd_mask operator&&(
            const simd_mask& x, const simd_mask& y) noexcept
        {
            return svand_z(x.all_true, x.pred, y.pred);
        }

        inline friend simd_mask operator||(
            const simd_mask& x, const simd_mask& y) noexcept
        {
            return svorr_z(x.all_true, x.pred, y.pred);
        }

        inline friend simd_mask operator&(
            const simd_mask& x, const simd_mask& y) noexcept
        {
            return svand_z(x.all_true, x.pred, y.pred);
        }

        inline friend simd_mask operator|(
            const simd_mask& x, const simd_mask& y) noexcept
        {
            return svorr_z(x.all_true, x.pred, y.pred);
        }

        inline friend simd_mask operator^(
            const simd_mask& x, const simd_mask& y) noexcept
        {
            return sveor_z(x.all_true, x.pred, y.pred);
        }

        // ----------------------------------------------------------------------
        // simd_mask compound assignment [simd.mask.cassign]
        // ----------------------------------------------------------------------
        inline friend simd_mask& operator&=(
            simd_mask& x, const simd_mask& y) noexcept
        {
            x.pred = svand_z(x.all_true, x.pred, y.pred);
            return x;
        }

        inline friend simd_mask& operator|=(
            simd_mask& x, const simd_mask& y) noexcept
        {
            x.pred = svorr_z(x.all_true, x.pred, y.pred);
            return x;
        }

        inline friend simd_mask& operator^=(
            simd_mask& x, const simd_mask& y) noexcept
        {
            x.pred = sveor_z(x.all_true, x.pred, y.pred);
            return x;
        }

        // ----------------------------------------------------------------------
        // simd_mask compares [simd.mask.comparison]
        // ----------------------------------------------------------------------
        inline friend simd_mask operator==(
            const simd_mask& x, const simd_mask& y) noexcept
        {
            return svnot_z(x.all_true, sveor_z(x.all_true, x.pred, y.pred));
        }

        inline friend simd_mask operator!=(
            const simd_mask& x, const simd_mask& y) noexcept
        {
            return sveor_z(x.all_true, x.pred, y.pred);
        }

        // ----------------------------------------------------------------------
        //  algorithms
        // ----------------------------------------------------------------------
        inline int popcount() const
        {
            return sve_impl::simd_impl_<T_size>::count(pred);
        }

        inline bool all_of() const
        {
            return popcount() == (int) size();
        }

        inline bool any_of() const
        {
            return popcount() > 0;
        }

        inline bool none_of() const
        {
            return popcount() != (int) size();
        }

        inline bool some_of() const
        {
            int c = popcount();
            return (c > 0) && (c < (int) size());
        }

        inline int find_first_set() const
        {
            auto index = sve_impl::simd_impl_<T_size>::first_true();
            for (int i = 0; i < (int) size(); i++)
            {
                if (sve_impl::simd_impl_<T_size>::count(
                        svand_z(all_true, pred, index)))
                    return i;
                index = sve_impl::simd_impl_<T_size>::next_true(index);
            }
            return -1;
        }

        inline int find_last_set() const
        {
            int ans = -1;
            auto index = sve_impl::simd_impl_<T_size>::first_true();
            for (int i = 0; i < (int) size(); i++)
            {
                if (sve_impl::simd_impl_<T_size>::count(
                        svand_z(all_true, pred, index)))
                    ans = i;
                index = sve_impl::simd_impl_<T_size>::next_true(index);
            }
            return ans;
        }

    private:
        template <typename T_, typename Abi_>
        inline friend simd<T_, Abi_> choose(const simd_mask<T_, Abi_>& msk,
            const simd<T_, Abi_>& t, const simd<T_, Abi_>& f);

        template <typename T_, typename Abi_>
        inline friend void mask_assign(const simd_mask<T_, Abi_>& msk,
            simd<T_, Abi_>& v, const simd<T_, Abi_>& val);

        template <typename T_, typename Abi_>
        inline friend simd<T_, Abi_> compact(
            const simd_mask<T_, Abi_>& msk, const simd<T_, Abi_>& v);

        template <typename T_, typename Abi_>
        inline friend simd<T_, Abi_> splice(const simd_mask<T_, Abi_>& msk,
            const simd<T_, Abi_>& x, const simd<T_, Abi_>& y);
    };

    template <class T, class Abi>
    inline bool all_of(const simd_mask<T, Abi>& m)
    {
        return m.all_of();
    }

    template <class T, class Abi>
    inline bool any_of(const simd_mask<T, Abi>& m)
    {
        return m.any_of();
    }

    template <class T, class Abi>
    inline bool none_of(const simd_mask<T, Abi>& m)
    {
        return m.none_of();
    }

    template <class T, class Abi>
    inline bool some_of(const simd_mask<T, Abi>& m)
    {
        return m.some_of();
    }

    template <class T, class Abi>
    inline int popcount(const simd_mask<T, Abi>& m)
    {
        return m.popcount();
    }

    template <class T, class Abi>
    inline int find_first_set(const simd_mask<T, Abi>& m)
    {
        return m.find_first_set();
    }

    template <class T, class Abi>
    inline int find_last_set(const simd_mask<T, Abi>& m)
    {
        return m.find_last_set();
    }

    template <typename T, typename Abi>
    inline simd<T, Abi> choose(const simd_mask<T, Abi>& msk,
        const simd<T, Abi>& t, const simd<T, Abi>& f)
    {
        return svsel(msk.pred, t.vec, f.vec);
    }

    template <typename T, typename Abi>
    inline void mask_assign(
        const simd_mask<T, Abi>& msk, simd<T, Abi>& v, const simd<T, Abi>& val)
    {
        v.vec = svsel(msk.pred, val.vec, v.vec);
    }

    template <typename T, typename Abi>
    inline simd<T, Abi> compact(
        const simd_mask<T, Abi>& msk, const simd<T, Abi>& v)
    {
        static_assert(sizeof(T) >= 4,
            "compact function only works with airthmetic types which\
         are atleast 4 bytes in size");
        return svcompact(msk.pred, v.vec);
    }

    template <typename T_, typename Abi_>
    inline simd<T_, Abi_> splice(const simd_mask<T_, Abi_>& msk,
        const simd<T_, Abi_>& x, const simd<T_, Abi_>& y)
    {
        return svsplice(msk.pred, x.vec, y.vec);
    }
}}    // namespace sve::experimental::parallelism_v2
#endif
