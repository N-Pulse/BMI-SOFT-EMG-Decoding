// emg_preprocessor.h
// Le pipeline complet, pour 8 canaux :
//   passe-haut -> passe-bas -> notch -> valeur absolue -> quantification -> fenêtres
// Même ordre que scripts/epn612/adaptation.py. Chaque étape peut être désactivée.
//
// Utilisation sur la carte :
//   emg::EmgPreprocessor pre;          // objet global ou static (~34 Ko), pas sur la pile
//   emg::PreprocessorConfig config;    // valeurs par défaut = celles de train.py
//   config.highpassEnabled = false;    // ex. : le passe-haut est fait en analogique
//   pre.init(config);
//
//   // dans l'interruption ADC, à 2000 Hz :
//   pre.pushAdcSample(adcCounts);
//
//   // dans la boucle principale :
//   if (pre.isWindowReady()) {
//       const float* window = pre.getWindow();   // forme (8, 400), canal par canal
//       ... donner la fenêtre au CNN ...
//       pre.clearWindowReady();
//   }
#ifndef EMG_PREPROCESSOR_H
#define EMG_PREPROCESSOR_H

#include "biquad.h"
#include "constants.h"
#include "quantizer.h"
#include "window_buffer.h"

#include <cstdint>

namespace emg {

// Tous les réglages. Les valeurs après "=" sont les valeurs par défaut (celles de train.py).
struct PreprocessorConfig {
    double samplingRateHz = 2000.0;
    int    channels       = 8;

    bool   highpassEnabled  = true;
    double highpassCutoffHz = 20.0;
    int    highpassOrder    = 1; //Plus on augmente, plus ca va diviser les fréquneces coupés brutalement

    bool   lowpassEnabled  = true;
    double lowpassCutoffHz = 500.0;
    int    lowpassOrder    = 1; //Plus on augmente, plus ca va diviser les fréquneces coupés brutalement

    bool   notchEnabled     = true;
    double notchFrequencyHz = 50.0;
    double notchQ           = 30.0; // notch width: bandwidth ≈ freq / Q (higher = narrower)


    bool   rectifyEnabled  = true;    // valeur absolue
    bool   quantizeEnabled = true;
    int    adcBits         = 12;
    int    adcOffset       = 0;       // "zéro" de l'ADC, soustrait dans pushAdcSample

    int    windowSize       = 400;
    int    windowStep       = 200;
    int    firstWindowStart = 0;
};

// Résultat de init(). "enum class" : on écrit ErrorCode::Ok, ErrorCode::BadWindow, etc.
enum class ErrorCode {
    Ok,
    BadSamplingRate,
    BadChannels,
    BadHighpass,
    BadLowpass,
    HighpassAboveLowpass,
    BadFilterOrder,
    BadNotch,
    BadAdcBits,
    BadWindowSize,
    BadWindowStep,
    BadFirstWindowStart
};

class EmgPreprocessor {
public:
    EmgPreprocessor();

    // Vérifie la config, calcule les coefficients, remet tout à zéro.
    ErrorCode init(const PreprocessorConfig& config);

    // Remet la mémoire des filtres et les fenêtres à zéro (garde la config).
    void reset();

    // Un échantillon par canal, déjà normalisé entre -1 et 1.
    void pushSample(const float frame[]);

    // Un échantillon par canal, en comptes ADC bruts.
    void pushAdcSample(const std::int16_t frame[]);

    // Plusieurs échantillons d'un coup, entrelacés : [c0 c1 .. c7][c0 c1 .. c7]...
    void pushBlock(const float frames[], int frameCount);

    // Passe UNE valeur d'UN canal dans la chaîne (sans fenêtrage). Utile pour les tests.
    float processValue(int channel, float x);

    bool isWindowReady() const;
    void clearWindowReady();
    const float* getWindow() const;
    long getWindowCount() const;

private:
    ErrorCode checkConfig(const PreprocessorConfig& config) const;

    PreprocessorConfig config_;

    // filters_[canal][section] : chaque canal a ses propres biquads (sa propre mémoire).
    Biquad filters_[MAX_CHANNELS][MAX_SECTIONS];
    int    sectionCount_;

    Quantizer    quantizer_;
    WindowBuffer windowBuffer_;

    // "volatile" : écrit dans l'interruption, lu dans la boucle principale.
    // Oblige le compilateur à relire la vraie valeur en mémoire à chaque fois.
    volatile bool windowReady_;
    long          windowCount_;
};

}  // namespace emg

#endif
