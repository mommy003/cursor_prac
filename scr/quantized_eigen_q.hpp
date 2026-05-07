//
//  quantized_eigen_q.hpp
//  SNP-column–scaled quantized Q: Q(j,i) ≈ q_int(j,i) * snpScales[i] / bound.
//  snpDequantScale[i] = snpScales[i]/bound is filled at load; MCMC uses explicit loops
//  with switch (bits) once per LD block (see model.cpp).
//

#ifndef quantized_eigen_q_hpp
#define quantized_eigen_q_hpp

#include "data.hpp"

inline int8_t quantizedEigenQNibbleToSigned4(unsigned n) {
    n &= 0x0Fu;
    return static_cast<int8_t>(static_cast<int8_t>(n << 4) >> 4);
}

/** Call once after reading raw + snpScales (before Model / wcorr init). */
inline void fillQuantizedEigenSnpDequantScale(QuantizedEigenQBlock &qb, int quantizedBits) {
    qb.snpDequantScale.resize(qb.m);
    float invB = 0.f;
    if (quantizedBits == 4) invB = 1.0f / 7.0f;
    else if (quantizedBits == 8) invB = 1.0f / 127.0f;
    else if (quantizedBits == 16) invB = 1.0f / 32767.0f;
    for (int i = 0; i < qb.m; ++i) {
        float s = qb.snpScales[i];
        qb.snpDequantScale[i] = (s > 0.f && invB > 0.f) ? s * invB : 0.f;
    }
}

#endif
