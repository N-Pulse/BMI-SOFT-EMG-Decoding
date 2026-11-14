// quantizer.h
// Simule l'ADC : arrondit le signal (entre -1 et 1) au niveau ADC le plus proche.
// Équivalent de quantize_signed_adc puis board_counts_to_normalized en Python.
#ifndef EMG_QUANTIZER_H
#define EMG_QUANTIZER_H

namespace emg {

class Quantizer {
public:
    Quantizer();

    // Exemple : 12 bits -> 4096 niveaux, comptes de -2048 à +2047.
    void setBits(int bits);

    // x -> round(x * 2048), borné à [-2048, 2047], puis / 2048.
    float process(float x) const;

private:
    float scale_;          // 2^(bits-1), par ex. 2048
    float inverseScale_;   // 1 / 2^(bits-1)
    float minCount_;       // -2^(bits-1)
    float maxCount_;       //  2^(bits-1) - 1
};

}  // namespace emg

#endif
