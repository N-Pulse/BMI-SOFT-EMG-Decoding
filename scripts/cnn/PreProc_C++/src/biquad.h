// biquad.h
// Un filtre IIR d'ordre 2 ("biquad"), pour UN canal.
//
//            b0 + b1 z^-1 + b2 z^-2
//   H(z) = --------------------------
//            1  + a1 z^-1 + a2 z^-2
//
// Un filtre d'ordre 1 est un biquad avec b2 = a2 = 0.
#ifndef EMG_BIQUAD_H
#define EMG_BIQUAD_H

namespace emg {

// Les 5 coefficients, calculés en double (précision max pendant la conception).
struct BiquadCoefficients {
    double b0;
    double b1;
    double b2;
    double a1;
    double a2;
};

class Biquad {
public:
    Biquad();

    // Copie les coefficients (convertis en float) et remet la mémoire à zéro.
    void setCoefficients(const BiquadCoefficients& coefficients);

    // Remet la mémoire du filtre à zéro (comme zi = 0 dans scipy).
    void reset();

    // Filtre UN échantillon et renvoie la sortie.
    float process(float input);

private:
    // Coefficients
    float b0_;
    float b1_;
    float b2_;
    float a1_;
    float a2_;

    // Mémoire du filtre entre deux échantillons (forme directe II transposée)
    float z1_;
    float z2_;
};

}  // namespace emg

#endif
