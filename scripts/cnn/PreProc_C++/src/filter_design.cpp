// filter_design.cpp
//
// Méthode (identique à scipy) :
//  1. Pôles du Butterworth analogique normalisé (coupure 1 rad/s) :
//       p_k = exp(j * pi * (2k + N + 1) / (2N)),  k = 0..N-1   (sur le cercle unité, partie réelle < 0)
//  2. Pré-déformation ("prewarping") de la coupure, pour que la fréquence de coupure
//     numérique tombe exactement sur cutoffHz :
//       Wc = 2 fs tan(pi fc / fs)
//  3. Mise à l'échelle :  passe-bas  s = Wc * p      passe-haut  s = Wc / p
//  4. Transformation bilinéaire (analogique -> numérique) :
//       z = (2fs + s) / (2fs - s)
//  5. Zéros : en z = -1 (passe-bas) ou z = +1 (passe-haut).
//  6. Gain réglé à 1 en DC (passe-bas) ou à Nyquist (passe-haut).
#include "filter_design.h"
#include "constants.h"

#include <cmath>
#include <complex>

namespace emg {

int designButterworth(int order, double cutoffHz, double samplingRateHz, bool isHighpass,
                      BiquadCoefficients sections[]) {
    const double k  = 2.0 * samplingRateHz;
    const double wc = k * std::tan(PI * cutoffHz / samplingRateHz);   // étape 2
    int sectionCount = 0;

    // --- Paires de pôles complexes conjugués -> un biquad par paire ---
    for (int i = 0; i < order / 2; ++i) {
        const double angle = PI * (2.0 * i + order + 1.0) / (2.0 * order);
        const std::complex<double> p(std::cos(angle), std::sin(angle));      // étape 1

        std::complex<double> s;                                               // étape 3
        if (isHighpass) {
            s = wc / p;
        } else {
            s = wc * p;
        }

        const std::complex<double> z = (k + s) / (k - s);                    // étape 4

        // Le pôle z et son conjugué donnent le dénominateur :
        //   (1 - z u)(1 - conj(z) u) = 1 - 2 Re(z) u + |z|^2 u^2   (u = z^-1)
        const double a1 = -2.0 * z.real();
        const double a2 = std::norm(z);   // norm = |z|^2

        BiquadCoefficients c;
        if (isHighpass) {
            // Zéros doubles en z = +1 : numérateur g (1 - 2u + u^2). Gain 1 en z = -1.
            const double g = (1.0 - a1 + a2) / 4.0;
            c.b0 = g;
            c.b1 = -2.0 * g;
            c.b2 = g;
        } else {
            // Zéros doubles en z = -1 : numérateur g (1 + 2u + u^2). Gain 1 en z = 1.
            const double g = (1.0 + a1 + a2) / 4.0;
            c.b0 = g;
            c.b1 = 2.0 * g;
            c.b2 = g;
        }
        c.a1 = a1;
        c.a2 = a2;
        sections[sectionCount] = c;
        sectionCount++;
    }

    // --- Ordre impair : il reste un pôle réel p = -1 -> section d'ordre 1 ---
    if (order % 2 == 1) {
        // s = -Wc (pour le passe-haut, Wc / (-1) donne aussi -Wc)
        const double z  = (k - wc) / (k + wc);
        const double a1 = -z;

        BiquadCoefficients c;
        if (isHighpass) {
            const double g = (1.0 - a1) / 2.0;   // numérateur g (1 - u), gain 1 en z = -1
            c.b0 = g;
            c.b1 = -g;
        } else {
            const double g = (1.0 + a1) / 2.0;   // numérateur g (1 + u), gain 1 en z = 1
            c.b0 = g;
            c.b1 = g;
        }
        c.b2 = 0.0;
        c.a1 = a1;
        c.a2 = 0.0;
        sections[sectionCount] = c;
        sectionCount++;
    }

    return sectionCount;
}

// Formule de scipy.signal.iirnotch :
//   w0 = 2 pi f0 / fs,  largeur de bande bw = w0 / Q
//   beta = tan(bw / 2),  gain = 1 / (1 + beta)
//   b = gain * [1, -2 cos(w0), 1]
//   a = [1, -2 gain cos(w0), 2 gain - 1]
BiquadCoefficients designNotch(double frequencyHz, double q, double samplingRateHz) {
    const double w0   = 2.0 * PI * frequencyHz / samplingRateHz;
    const double bw   = w0 / q;
    const double beta = std::tan(bw / 2.0);
    const double gain = 1.0 / (1.0 + beta);

    BiquadCoefficients c;
    c.b0 = gain;
    c.b1 = -2.0 * gain * std::cos(w0);
    c.b2 = gain;
    c.a1 = -2.0 * gain * std::cos(w0);
    c.a2 = 2.0 * gain - 1.0;
    return c;
}

}  // namespace emg
