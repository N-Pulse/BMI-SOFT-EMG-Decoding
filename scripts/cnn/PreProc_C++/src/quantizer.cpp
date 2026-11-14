// quantizer.cpp
#include "quantizer.h"

#include <cmath>

namespace emg {

Quantizer::Quantizer() {
    setBits(12);
}

void Quantizer::setBits(int bits) {
    // std::ldexp(1.0, n) = 1.0 * 2^n
    const double half = std::ldexp(1.0, bits - 1);
    scale_        = static_cast<float>(half);
    inverseScale_ = static_cast<float>(1.0 / half);
    minCount_     = static_cast<float>(-half);
    maxCount_     = static_cast<float>(half - 1.0);
}

float Quantizer::process(float x) const {
    // std::nearbyint arrondit "au pair le plus proche" en cas d'égalité (2.5 -> 2, 3.5 -> 4),
    // exactement comme numpy.rint. (std::round, lui, ferait 2.5 -> 3.)
    float counts = std::nearbyint(x * scale_);

    if (counts < minCount_) {
        counts = minCount_;
    }
    if (counts > maxCount_) {
        counts = maxCount_;
    }

    // Multiplier par 1/2048 est exact (puissance de 2) et plus rapide qu'une division.
    return counts * inverseScale_;
}

}  // namespace emg
