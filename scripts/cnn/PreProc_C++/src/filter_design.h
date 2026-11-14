// filter_design.h
// Calcul des coefficients des filtres (fait une seule fois, à l'initialisation).
#ifndef EMG_FILTER_DESIGN_H
#define EMG_FILTER_DESIGN_H

#include "biquad.h"

namespace emg {

// Butterworth passe-haut ou passe-bas, équivalent à scipy.signal.butter(..., output="sos").
// Écrit les biquads dans le tableau "sections" et renvoie combien il en a écrit.
int designButterworth(int order, double cutoffHz, double samplingRateHz, bool isHighpass,
                      BiquadCoefficients sections[]);

// Notch (coupe-bande étroit), équivalent à scipy.signal.iirnotch.
BiquadCoefficients designNotch(double frequencyHz, double q, double samplingRateHz);

}  // namespace emg

#endif
