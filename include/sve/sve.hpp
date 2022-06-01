#pragma once

#ifdef __ARM_FEATURE_SVE

#include <arm_sve.h>
#include <cstddef>
#include <functional>
#include <iostream>
#include <type_traits>

namespace sve_impl {
    template <typename T>
    struct simd_impl
    {
    };

    static constexpr int max_vector_pack_size = 64;
    typedef svbool_t Predicate __attribute__((arm_sve_vector_bits(512)));

    // ----------------------------------------------------------------------
    template <>
    struct simd_impl<int8_t>
    {
        typedef svint8_t Vector __attribute__((arm_sve_vector_bits(512)));
        static constexpr std::size_t size =
            max_vector_pack_size / sizeof(int8_t);

        inline static constexpr auto pred_all_true = svptrue_b8;
        inline static constexpr auto pred_next = svpnext_b8;
        inline static constexpr auto pred_pat_true = svptrue_pat_b8;
        inline static constexpr auto pred_count = svcntp_b8;
        inline static constexpr auto set_helper = svdup_s8_m;

        static inline Vector fill(int8_t val)
        {
            return svdup_s8(val);
        }
    };

    template <>
    struct simd_impl<uint8_t>
    {
        typedef svuint8_t Vector __attribute__((arm_sve_vector_bits(512)));
        static constexpr std::size_t size =
            max_vector_pack_size / sizeof(uint8_t);

        inline static constexpr auto pred_all_true = svptrue_b8;
        inline static constexpr auto pred_next = svpnext_b8;
        inline static constexpr auto pred_pat_true = svptrue_pat_b8;
        inline static constexpr auto pred_count = svcntp_b8;
        inline static constexpr auto set_helper = svdup_u8_m;

        static inline Vector fill(uint8_t val)
        {
            return svdup_u8(val);
        }
    };

    template <>
    struct simd_impl<int16_t>
    {
        typedef svint16_t Vector __attribute__((arm_sve_vector_bits(512)));
        static constexpr std::size_t size =
            max_vector_pack_size / sizeof(int16_t);

        inline static constexpr auto pred_all_true = svptrue_b16;
        inline static constexpr auto pred_next = svpnext_b16;
        inline static constexpr auto pred_pat_true = svptrue_pat_b16;
        inline static constexpr auto pred_count = svcntp_b16;
        inline static constexpr auto set_helper = svdup_s16_m;

        static inline Vector fill(int16_t val)
        {
            return svdup_s16(val);
        }
    };

    template <>
    struct simd_impl<uint16_t>
    {
        typedef svuint16_t Vector __attribute__((arm_sve_vector_bits(512)));
        static constexpr std::size_t size =
            max_vector_pack_size / sizeof(uint16_t);

        inline static constexpr auto pred_all_true = svptrue_b16;
        inline static constexpr auto pred_next = svpnext_b16;
        inline static constexpr auto pred_pat_true = svptrue_pat_b16;
        inline static constexpr auto pred_count = svcntp_b16;
        inline static constexpr auto set_helper = svdup_u16_m;

        static inline Vector fill(uint16_t val)
        {
            return svdup_u16(val);
        }
    };

    template <>
    struct simd_impl<int32_t>
    {
        typedef svint32_t Vector __attribute__((arm_sve_vector_bits(512)));
        static constexpr std::size_t size =
            max_vector_pack_size / sizeof(int32_t);

        inline static constexpr auto pred_all_true = svptrue_b32;
        inline static constexpr auto pred_next = svpnext_b32;
        inline static constexpr auto pred_pat_true = svptrue_pat_b32;
        inline static constexpr auto pred_count = svcntp_b32;
        inline static constexpr auto set_helper = svdup_s32_m;

        static inline Vector fill(int32_t val)
        {
            return svdup_s32(val);
        }
    };

    template <>
    struct simd_impl<uint32_t>
    {
        typedef svuint32_t Vector __attribute__((arm_sve_vector_bits(512)));
        static constexpr std::size_t size =
            max_vector_pack_size / sizeof(uint32_t);

        inline static constexpr auto pred_all_true = svptrue_b32;
        inline static constexpr auto pred_next = svpnext_b32;
        inline static constexpr auto pred_pat_true = svptrue_pat_b32;
        inline static constexpr auto pred_count = svcntp_b32;
        inline static constexpr auto set_helper = svdup_u32_m;

        static inline Vector fill(uint32_t val)
        {
            return svdup_u32(val);
        }
    };

    template <>
    struct simd_impl<int64_t>
    {
        typedef svint64_t Vector __attribute__((arm_sve_vector_bits(512)));
        static constexpr std::size_t size =
            max_vector_pack_size / sizeof(int64_t);

        inline static constexpr auto pred_all_true = svptrue_b64;
        inline static constexpr auto pred_next = svpnext_b64;
        inline static constexpr auto pred_pat_true = svptrue_pat_b64;
        inline static constexpr auto pred_count = svcntp_b64;
        inline static constexpr auto set_helper = svdup_s64_m;

        static inline Vector fill(int64_t val)
        {
            return svdup_s64(val);
        }
    };

    template <>
    struct simd_impl<uint64_t>
    {
        typedef svuint64_t Vector __attribute__((arm_sve_vector_bits(512)));
        static constexpr std::size_t size =
            max_vector_pack_size / sizeof(uint64_t);

        inline static constexpr auto pred_all_true = svptrue_b64;
        inline static constexpr auto pred_next = svpnext_b64;
        inline static constexpr auto pred_pat_true = svptrue_pat_b64;
        inline static constexpr auto pred_count = svcntp_b64;
        inline static constexpr auto set_helper = svdup_u64_m;

        static inline Vector fill(uint64_t val)
        {
            return svdup_u64(val);
        }
    };

    // ----------------------------------------------------------------------
    template <>
    struct simd_impl<float16_t>
    {
        typedef svfloat16_t Vector __attribute__((arm_sve_vector_bits(512)));
        static constexpr std::size_t size =
            max_vector_pack_size / sizeof(float16_t);

        inline static constexpr auto pred_all_true = svptrue_b16;
        inline static constexpr auto pred_next = svpnext_b16;
        inline static constexpr auto pred_pat_true = svptrue_pat_b16;
        inline static constexpr auto pred_count = svcntp_b16;
        inline static constexpr auto set_helper = svdup_f16_m;

        static inline Vector fill(float16_t val)
        {
            return svdup_f16(val);
        }
    };

    template <>
    struct simd_impl<float>
    {
        typedef svfloat32_t Vector __attribute__((arm_sve_vector_bits(512)));
        static constexpr std::size_t size =
            max_vector_pack_size / sizeof(float);

        inline static constexpr auto pred_all_true = svptrue_b32;
        inline static constexpr auto pred_next = svpnext_b32;
        inline static constexpr auto pred_pat_true = svptrue_pat_b32;
        inline static constexpr auto pred_count = svcntp_b32;
        inline static constexpr auto set_helper = svdup_f32_m;

        static inline Vector fill(float val)
        {
            return svdup_f32(val);
        }
    };

    template <>
    struct simd_impl<double>
    {
        typedef svfloat64_t Vector __attribute__((arm_sve_vector_bits(512)));
        static constexpr std::size_t size =
            max_vector_pack_size / sizeof(double);

        inline static constexpr auto pred_all_true = svptrue_b64;
        inline static constexpr auto pred_next = svpnext_b64;
        inline static constexpr auto pred_pat_true = svptrue_pat_b64;
        inline static constexpr auto pred_count = svcntp_b64;
        inline static constexpr auto set_helper = svdup_f64_m;

        static inline Vector fill(double val)
        {
            return svdup_f64(val);
        }
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
        Predicate all_true;

        inline void add(T val)
        {
            vec = svadd_x(all_true, vec, val);
        }

        inline void add(Vector val)
        {
            vec = svadd_x(all_true, vec, val);
        }

        static inline Vector add(Vector val1, Vector val2)
        {
            return svadd_x(sve_impl::simd_impl<T>::pred_all_true(), val1, val2);
        }

        inline void subtract(T val)
        {
            vec = svsub_x(all_true, vec, val);
        }

        inline void subtract(Vector val)
        {
            vec = svsub_x(all_true, vec, val);
        }

        static inline Vector subtract(Vector val1, Vector val2)
        {
            return svsub_x(sve_impl::simd_impl<T>::pred_all_true(), val1, val2);
        }

        inline void multiply(T val)
        {
            vec = svmul_x(all_true, vec, val);
        }

        inline void multiply(Vector val)
        {
            vec = svmul_x(all_true, vec, val);
        }

        static inline Vector multiply(Vector val1, Vector val2)
        {
            return svmul_x(sve_impl::simd_impl<T>::pred_all_true(), val1, val2);
        }

        static inline Vector divide(Vector val1, Vector val2)
        {
            return svdiv_x(sve_impl::simd_impl<T>::pred_all_true(), val1, val2);
        }

        static inline Vector bitwise_and(Vector val1, Vector val2)
        {
            return svand_x(sve_impl::simd_impl<T>::pred_all_true(), val1, val2);
        }

        static inline Vector bitwise_or(Vector val1, Vector val2)
        {
            return svorr_x(sve_impl::simd_impl<T>::pred_all_true(), val1, val2);
        }

        static inline Vector bitwise_xor(Vector val1, Vector val2)
        {
            return sveor_x(sve_impl::simd_impl<T>::pred_all_true(), val1, val2);
        }

        template <typename U>
        static inline Vector left_shift(Vector val1, U val2)
        {
            return svlsl_x(sve_impl::simd_impl<T>::pred_all_true(), val1, val2);
        }

        template <typename U>
        static inline Vector right_shift(Vector val1, U val2)
        {
            return svlsr_x(sve_impl::simd_impl<T>::pred_all_true(), val1, val2);
        }

    public:
        using value_type = T;
        using abi_type = Abi;
        using mask_type = simd_mask<T, Abi>;

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
            all_true = sve_impl::simd_impl<T>::pred_all_true();
            vec = svld1(all_true, ptr);
        }

        // template <typename U>
        inline simd(T val = {})
        {
            vec = sve_impl::simd_impl<T>::fill(val);
            all_true = sve_impl::simd_impl<T>::pred_all_true();
        }

        inline simd(Vector v)
        {
            vec = v;
            all_true = sve_impl::simd_impl<T>::pred_all_true();
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

            auto index = sve_impl::simd_impl<T>::pred_pat_true(SV_VL1);
            for (int i = 0; i < idx; i++)
            {
                index = sve_impl::simd_impl<T>::pred_next(all_true, index);
            }
            return svlastb(index, vec);
        }

        T operator[](int idx) const
        {
            return get(idx);
        }

        void set(int idx, T val)
        {
            if (idx < 0 || idx > (int) size())
                return;

            auto index = sve_impl::simd_impl<T>::pred_pat_true(SV_VL1);
            for (int i = 0; i < idx; i++)
            {
                index = sve_impl::simd_impl<T>::pred_next(all_true, index);
            }
            vec = sve_impl::simd_impl<T>::set_helper(vec, index, val);
        }

        // ----------------------------------------------------------------------
        // ostream overload
        // ----------------------------------------------------------------------
        friend std::ostream& operator<<(std::ostream& os, const simd& x)
        {
            using type_ = std::decay_t<decltype(x)>;
            using value_type_ = typename type_::value_type;

            auto index =
                sve_impl::simd_impl<value_type_>::pred_pat_true(SV_VL1);
            os << "( ";
            for (int i = 0; i < (int) x.size(); i++)
            {
                os << svlastb(index, x.vec) << ' ';
                index = sve_impl::simd_impl<value_type_>::pred_next(
                    x.all_true, index);
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
            add(static_cast<T>(1));
            return *this;
        }

        inline auto operator++(int)
        {
            auto vec_copy = *this;
            add(static_cast<T>(1));
            return vec_copy;
        }

        inline simd& operator--()
        {
            subtract(static_cast<T>(1));
            return *this;
        }

        inline auto operator--(int)
        {
            auto vec_copy = *this;
            subtract(static_cast<T>(1));
            return vec_copy;
        }

        inline simd operator+() const
        {
            return *this;
        }

        inline simd operator-() const
        {
            auto vec_copy = *this;
            vec_copy.multiply(-1);
            return vec_copy;
        }

        // ----------------------------------------------------------------------
        // binary operators [simd.binary]
        // ----------------------------------------------------------------------
        inline friend simd operator+(const simd& x, const simd& y)
        {
            return add(x.vec, y.vec);
        }

        inline friend simd operator-(const simd& x, const simd& y)
        {
            return subtract(x.vec, y.vec);
        }

        inline friend simd operator*(const simd& x, const simd& y)
        {
            return multiply(x.vec, y.vec);
        }

        inline friend simd operator/(const simd& x, const simd& y)
        {
            return divide(x.vec, y.vec);
        }

        inline friend simd operator&(const simd& x, const simd& y)
        {
            static_assert(std::is_integral_v<T>,
                "operator& only works for integeral types");
            return bitwise_and(x.vec, y.vec);
        }

        inline friend simd operator|(const simd& x, const simd& y)
        {
            static_assert(std::is_integral_v<T>,
                "operator| only works for integeral types");
            return bitwise_or(x.vec, y.vec);
        }

        inline friend simd operator^(const simd& x, const simd& y)
        {
            static_assert(std::is_integral_v<T>,
                "operator^ only works for integeral types");
            return bitwise_xor(x.vec, y.vec);
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
            x.vec = add(x.vec, y.vec);
            return x;
        }

        inline friend simd& operator-=(simd& x, const simd& y)
        {
            x.vec = subtract(x.vec, y.vec);
            return x;
        }

        inline friend simd& operator*=(simd& x, const simd& y)
        {
            x.vec = multiply(x.vec, y.vec);
            return x;
        }

        inline friend simd& operator/=(simd& x, const simd& y)
        {
            x.vec = divide(x.vec, y.vec);
            return x;
        }

        inline friend simd operator&=(simd& x, const simd& y)
        {
            static_assert(std::is_integral_v<T>,
                "operator&= only works for integeral types");
            x.vec = bitwise_and(x.vec, y.vec);
            return x;
        }

        inline friend simd operator|=(simd& x, const simd& y)
        {
            static_assert(std::is_integral_v<T>,
                "operator|= only works for integeral types");
            x.vec = bitwise_or(x.vec, y.vec);
            return x;
        }

        inline friend simd operator^=(simd& x, const simd& y)
        {
            static_assert(std::is_integral_v<T>,
                "operator^= only works for integeral types");
            x.vec = bitwise_xor(x.vec, y.vec);
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
        inline friend simd<T_, Abi_> min(
            const simd<T_, Abi_>& x, const simd<T_, Abi_>& y);

        template <typename T_, typename Abi_>
        inline friend simd<T_, Abi_> max(
            const simd<T_, Abi_>& x, const simd<T_, Abi_>& y);

        // template <typename T_, typename Abi_>
        // inline friend T_ reduce(const simd<T_, Abi_>& x, std::plus<>)
        // {
        //     return svaddv(x.all_true, x.vec);
        // }

        // template <typename T_, typename Abi_>
        // inline friend T reduce(const simd<T_, Abi_>& x, std::bit_and<>)
        // {
        //     static_assert(std::is_integral_v<T_>,
        //         "bit_and reduction only works this integeral types");
        //     return svandv(x.all_true, x.vec);
        // }

        // template <typename T_, typename Abi_>
        // inline friend T_ reduce(const simd<T_, Abi_>& x, std::bit_or<>)
        // {
        //     static_assert(std::is_integral_v<T_>,
        //         "bit_or reduction only works this integeral types");
        //     return svorv(x.all_true, x.vec);
        // }

        // template <typename T_, typename Abi_>
        // inline friend T_ reduce(const simd<T_, Abi_>& x, std::bit_xor<>)
        // {
        //     static_assert(std::is_integral_v<T_>,
        //         "bit_xor reduction only works this integeral types");
        //     return sveorv(x.all_true, x.vec);
        // }

        template <typename T_, typename Abi_>
        inline friend simd<T_, Abi_> sqrt(const simd<T_, Abi_>& x);
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
    inline simd<T_, Abi_> sqrt(const simd<T_, Abi_>& x)
    {
        static_assert(std::is_floating_point_v<T_> ||
                    std::is_same_v<T_, float16_t>,
                    "sqrt only works this floating point types");
        return svsqrt_x(x.all_true, x.vec);
    }

    template <typename T, typename Abi>
    class simd_mask
    {
    private:
        using Predicate = typename sve_impl::Predicate;
        Predicate pred, all_true;

    public:
        using value_type = bool;
        using simd_type = simd<T, Abi>;
        using abi_type = Abi;

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
            all_true = sve_impl::simd_impl<T>::pred_all_true();
            if (val)
                pred = sve_impl::simd_impl<T>::pred_all_true();
            else
                pred = svpfalse();
        }

        inline simd_mask(Predicate p)
        {
            all_true = sve_impl::simd_impl<T>::pred_all_true();
            pred = p;
        }

        // ----------------------------------------------------------------------
        //  get and set
        // ----------------------------------------------------------------------
        int get(int idx) const
        {
            if (idx < 0 || idx > (int) size())
                return -1;

            auto index = sve_impl::simd_impl<T>::pred_pat_true(SV_VL1);
            for (int i = 0; i < idx; i++)
            {
                index = sve_impl::simd_impl<T>::pred_next(all_true, index);
            }
            index = svand_z(all_true, pred, index);
            return sve_impl::simd_impl<T>::pred_count(all_true, index);
        }

        T operator[](int idx) const
        {
            return get(idx);
        }

        void set(int idx, bool val)
        {
            if (idx < 0 || idx > (int) size())
                return;

            auto index = sve_impl::simd_impl<T>::pred_pat_true(SV_VL1);
            for (int i = 0; i < idx; i++)
            {
                index = sve_impl::simd_impl<T>::pred_next(all_true, index);
            }
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

            auto index =
                sve_impl::simd_impl<value_type_>::pred_pat_true(SV_VL1);
            os << "( ";
            for (int i = 0; i < sve_impl::simd_impl<value_type_>::size; i++)
            {
                os << sve_impl::simd_impl<value_type_>::pred_count(
                          x.all_true, svand_z(x.all_true, x.pred, index))
                   << ' ';
                index = sve_impl::simd_impl<value_type_>::pred_next(
                    x.all_true, index);
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
            return sve_impl::simd_impl<T>::pred_count(all_true, pred);
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
            auto index = sve_impl::simd_impl<T>::pred_pat_true(SV_VL1);
            for (int i = 0; i < (int) size(); i++)
            {
                if (sve_impl::simd_impl<T>::pred_count(
                        all_true, svand_z(all_true, pred, index)))
                    return i;
                index = sve_impl::simd_impl<T>::pred_next(all_true, index);
            }
            return -1;
        }

        inline int find_last_set() const
        {
            int ans = -1;
            auto index = sve_impl::simd_impl<T>::pred_pat_true(SV_VL1);
            for (int i = 0; i < (int) size(); i++)
            {
                if (sve_impl::simd_impl<T>::pred_count(
                        all_true, svand_z(all_true, pred, index)))
                    ans = i;
                index = sve_impl::simd_impl<T>::pred_next(all_true, index);
            }
            return ans;
        }

    private:
        template <typename T_, typename Abi_>
        inline friend simd<T_, Abi_> choose(const simd_mask<T_, Abi_>& msk,
            const simd<T_, Abi_>& t, const simd<T_, Abi_>& f);
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
    inline void mask_assign(const simd_mask<T, Abi>& msk,
        simd<T, Abi>& v, const simd<T, Abi>& val)
    {
        v.vec = svsel(msk.pred, val.vec, v.vec);
    }
}}    // namespace sve::experimental::parallelism_v2
#endif
