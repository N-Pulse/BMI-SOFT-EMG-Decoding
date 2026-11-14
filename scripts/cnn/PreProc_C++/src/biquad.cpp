// biquad.cpp
#include "biquad.h"

namespace emg {

//------ Role: create a pass-through filter (output = input) with empty memory ------
Biquad::Biquad()
    : b0_(1.0f), b1_(0.0f), b2_(0.0f), a1_(0.0f), a2_(0.0f), z1_(0.0f), z2_(0.0f) {
    // Par défaut : filtre "identité" (sortie = entrée).
}

//------ Role: turn this generic biquad into a specific filter (HP, LP or notch) ------
void Biquad::setCoefficients(const BiquadCoefficients& coefficients) {
    b0_ = static_cast<float>(coefficients.b0);
    b1_ = static_cast<float>(coefficients.b1);
    b2_ = static_cast<float>(coefficients.b2);
    a1_ = static_cast<float>(coefficients.a1);
    a2_ = static_cast<float>(coefficients.a2);
    reset();
}

//------ Role: clear the filter memory (same as scipy's default zi = 0) ------
void Biquad::reset() {
    z1_ = 0.0f;
    z2_ = 0.0f;
}

//------ Role: filter ONE sample -> return ONE filtered sample ------
// Forme directe II transposée (même structure que scipy.signal.sosfilt / lfilter) :
//   y[n]  = b0 x[n] + z1
//   z1    = b1 x[n] - a1 y[n] + z2
//   z2    = b2 x[n] - a2 y[n]
float Biquad::process(float input) {
    const float output = b0_ * input + z1_;
    z1_ = b1_ * input - a1_ * output + z2_;
    z2_ = b2_ * input - a2_ * output;
    return output;
}

}  // namespace emg

