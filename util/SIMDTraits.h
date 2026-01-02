#ifndef SIMD_TRAITS_H
#define SIMD_TRAITS_H

/**
 * @file SIMDTraits.h
 * @brief Compile-time SIMD trait definitions
 *
 * Defines type traits and compile-time constants for different SIMD levels
 */

// Forward declarations
enum class SIMDLevel {
    SCALAR,  // No SIMD
    SSE,     // 128-bit, 2 doubles
    AVX,     // 256-bit, 4 doubles
    AVX2,    // 256-bit + FMA
    AVX512   // 512-bit, 8 doubles
};

/**
 * @brief SIMD register width in number of doubles
 *
 * Template specialization for different SIMD levels
 */
template<SIMDLevel Level>
struct SIMDWidth {
    static constexpr int value = 1;  // Default scalar
};

// Specializations for each SIMD level
template<>
struct SIMDWidth<SIMDLevel::SSE> {
    static constexpr int value = 2;  // SSE: 2 doubles (128-bit)
};

template<>
struct SIMDWidth<SIMDLevel::AVX> {
    static constexpr int value = 4;  // AVX: 4 doubles (256-bit)
};

template<>
struct SIMDWidth<SIMDLevel::AVX2> {
    static constexpr int value = 4;  // AVX2: 4 doubles (256-bit) + FMA
};

template<>
struct SIMDWidth<SIMDLevel::AVX512> {
    static constexpr int value = 8;  // AVX-512: 8 doubles (512-bit)
};

/**
 * @brief Memory alignment requirements for SIMD levels
 *
 * Different SIMD instructions require different alignment for optimal performance
 */
template<SIMDLevel Level>
struct SIMDAlignment {
    static constexpr int value = alignof(double);  // Default alignment
};

template<>
struct SIMDAlignment<SIMDLevel::AVX> {
    static constexpr int value = 32;  // AVX requires 32-byte alignment
};

template<>
struct SIMDAlignment<SIMDLevel::AVX2> {
    static constexpr int value = 32;  // AVX2 requires 32-byte alignment
};

template<>
struct SIMDAlignment<SIMDLevel::AVX512> {
    static constexpr int value = 64;  // AVX-512 requires 64-byte alignment
};

/**
 * @brief Helper to get SIMD width at runtime
 *
 * @param level SIMD level
 * @return Number of doubles that fit in one SIMD register
 */
inline int getSIMDWidth(SIMDLevel level) {
    switch (level) {
        case SIMDLevel::AVX512: return SIMDWidth<SIMDLevel::AVX512>::value;
        case SIMDLevel::AVX2:   return SIMDWidth<SIMDLevel::AVX2>::value;
        case SIMDLevel::AVX:    return SIMDWidth<SIMDLevel::AVX>::value;
        case SIMDLevel::SSE:    return SIMDWidth<SIMDLevel::SSE>::value;
        case SIMDLevel::SCALAR: return SIMDWidth<SIMDLevel::SCALAR>::value;
        default:                return 1;
    }
}

/**
 * @brief Helper to get alignment requirement at runtime
 *
 * @param level SIMD level
 * @return Required alignment in bytes
 */
inline int getSIMDAlignment(SIMDLevel level) {
    switch (level) {
        case SIMDLevel::AVX512: return SIMDAlignment<SIMDLevel::AVX512>::value;
        case SIMDLevel::AVX2:   return SIMDAlignment<SIMDLevel::AVX2>::value;
        case SIMDLevel::AVX:    return SIMDAlignment<SIMDLevel::AVX>::value;
        case SIMDLevel::SSE:    return SIMDAlignment<SIMDLevel::SSE>::value;
        case SIMDLevel::SCALAR: return SIMDAlignment<SIMDLevel::SCALAR>::value;
        default:                return alignof(double);
    }
}

#endif // SIMD_TRAITS_H
