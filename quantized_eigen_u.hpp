//
//  quantized_eigen_u.hpp
//  Quantized eigenvector matrix U (m×k), row-major raw layout matches Q-per-SNP storage:
//  column SNP i is contiguous: index i*k + j = U_{i,j}.  Q_{j,i} = sqrt(lambda[j])*U_{i,j}.
//  sqrtLambdaScaleDequant[j] = Q dequant factor for column j.
//  If scales are U-column scales: sqrt(lambda[j]) * eigenColScale[j] / quantBound.
//  If scales already include sqrt(lambda): eigenColScale[j] / quantBound.
//

#ifndef quantized_eigen_u_hpp
#define quantized_eigen_u_hpp

#include "data.hpp"

inline void fillQuantizedEigenSqrtLambdaScaleDequant(QuantizedEigenUBlock &ub, int quantizedBits) {
    float invB = 0.f;
    if (quantizedBits == 4) invB = 1.0f / 7.0f;
    else if (quantizedBits == 8) invB = 1.0f / 127.0f;
    else if (quantizedBits == 16) invB = 1.0f / 32767.0f;
    ub.sqrtLambdaScaleDequant.resize(ub.k);
    for (int j = 0; j < ub.k; ++j) {
        float lam = ub.lambda[j];
        float s = ub.eigenScales[j];
        float sq = 0.f;
        if (s > 0.f && invB > 0.f) {
            if (ub.scalesIncludeSqrtLambda) {
                sq = s * invB;
            } else if (lam > 0.f) {
                sq = sqrtf(lam) * s * invB;
            }
        }
        ub.sqrtLambdaScaleDequant[j] = sq;
    }
}

#endif
