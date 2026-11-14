// emg_preprocessor.cpp
#include "emg_preprocessor.h"
#include "filter_design.h"

#include <cmath>

namespace emg {

EmgPreprocessor::EmgPreprocessor() : sectionCount_(0), windowReady_(false), windowCount_(0) {
    init(PreprocessorConfig());   // configuration par défaut
}

// Mêmes vérifications que BoardAdaptationConfig.__post_init__ en Python.
ErrorCode EmgPreprocessor::checkConfig(const PreprocessorConfig& c) const {
    const double nyquist = c.samplingRateHz / 2.0;

    if (c.samplingRateHz <= 0.0) return ErrorCode::BadSamplingRate;
    if (c.channels < 1 || c.channels > MAX_CHANNELS) return ErrorCode::BadChannels;

    if (c.highpassEnabled) {
        if (c.highpassOrder < 1 || c.highpassOrder > MAX_ORDER) return ErrorCode::BadFilterOrder;
        if (c.highpassCutoffHz <= 0.0 || c.highpassCutoffHz >= nyquist) return ErrorCode::BadHighpass;
    }
    if (c.lowpassEnabled) {
        if (c.lowpassOrder < 1 || c.lowpassOrder > MAX_ORDER) return ErrorCode::BadFilterOrder;
        if (c.lowpassCutoffHz <= 0.0 || c.lowpassCutoffHz >= nyquist) return ErrorCode::BadLowpass;
    }
    if (c.highpassEnabled && c.lowpassEnabled && c.highpassCutoffHz >= c.lowpassCutoffHz) {
        return ErrorCode::HighpassAboveLowpass;
    }
    if (c.notchEnabled) {
        if (c.notchFrequencyHz <= 0.0 || c.notchFrequencyHz >= nyquist || c.notchQ <= 0.0) {
            return ErrorCode::BadNotch;
        }
    }

    if (c.adcBits < 2 || c.adcBits > 24) return ErrorCode::BadAdcBits;
    if (c.windowSize < 1 || c.windowSize > MAX_WINDOW) return ErrorCode::BadWindowSize;
    if (c.windowStep < 1) return ErrorCode::BadWindowStep;
    if (c.firstWindowStart < 0) return ErrorCode::BadFirstWindowStart;

    return ErrorCode::Ok;
}

ErrorCode EmgPreprocessor::init(const PreprocessorConfig& config) {
    const ErrorCode error = checkConfig(config);
    if (error != ErrorCode::Ok) {
        return error;
    }
    config_ = config;

    // 1. Calcul des coefficients, dans l'ordre de la chaîne : HP, LP, notch.
    BiquadCoefficients sections[MAX_SECTIONS];
    sectionCount_ = 0;

    if (config.highpassEnabled) {
        sectionCount_ += designButterworth(config.highpassOrder, config.highpassCutoffHz,
                                           config.samplingRateHz, true, &sections[sectionCount_]);
    }
    if (config.lowpassEnabled) {
        sectionCount_ += designButterworth(config.lowpassOrder, config.lowpassCutoffHz,
                                           config.samplingRateHz, false, &sections[sectionCount_]);
    }
    if (config.notchEnabled) {
        sections[sectionCount_] = designNotch(config.notchFrequencyHz, config.notchQ,
                                              config.samplingRateHz);
        sectionCount_++;
    }

    // 2. Mêmes coefficients pour chaque canal (chaque canal garde sa propre mémoire).
    for (int channel = 0; channel < MAX_CHANNELS; ++channel) {
        for (int s = 0; s < sectionCount_; ++s) {
            filters_[channel][s].setCoefficients(sections[s]);
        }
    }

    // 3. Quantificateur et fenêtres.
    quantizer_.setBits(config.adcBits);
    windowBuffer_.configure(config.channels, config.windowSize, config.windowStep,
                            config.firstWindowStart);

    reset();
    return ErrorCode::Ok;
}

void EmgPreprocessor::reset() {
    for (int channel = 0; channel < MAX_CHANNELS; ++channel) {
        for (int s = 0; s < MAX_SECTIONS; ++s) {
            filters_[channel][s].reset();
        }
    }
    windowBuffer_.reset();
    windowReady_ = false;
    windowCount_ = 0;
}

float EmgPreprocessor::processValue(int channel, float x) {
    // HP -> LP -> notch : on passe dans chaque biquad l'un après l'autre.
    for (int s = 0; s < sectionCount_; ++s) {
        x = filters_[channel][s].process(x);
    }
    if (config_.rectifyEnabled) {
        x = std::fabs(x);
    }
    if (config_.quantizeEnabled) {
        x = quantizer_.process(x);
    }
    return x;
}

void EmgPreprocessor::pushSample(const float frame[]) {
    float processed[MAX_CHANNELS];
    for (int channel = 0; channel < config_.channels; ++channel) {
        processed[channel] = processValue(channel, frame[channel]);
    }

    const bool newWindow = windowBuffer_.push(processed);
    if (newWindow) {
        windowReady_ = true;
        windowCount_++;
    }
}

void EmgPreprocessor::pushAdcSample(const std::int16_t frame[]) {
    // Conversion comptes ADC -> valeur entre -1 et 1 : (comptes - offset) / 2^(bits-1)
    const float scale = static_cast<float>(1.0 / std::ldexp(1.0, config_.adcBits - 1));
    float normalized[MAX_CHANNELS];
    for (int channel = 0; channel < config_.channels; ++channel) {
        normalized[channel] = static_cast<float>(frame[channel] - config_.adcOffset) * scale;
    }
    pushSample(normalized);
}

void EmgPreprocessor::pushBlock(const float frames[], int frameCount) {
    // Juste une boucle sur pushSample : le résultat est donc identique,
    // qu'on envoie les échantillons un par un ou par blocs.
    for (int i = 0; i < frameCount; ++i) {
        pushSample(&frames[i * config_.channels]);
    }
}

bool EmgPreprocessor::isWindowReady() const {
    return windowReady_;
}

void EmgPreprocessor::clearWindowReady() {
    windowReady_ = false;
}

const float* EmgPreprocessor::getWindow() const {
    return windowBuffer_.getWindow();
}

long EmgPreprocessor::getWindowCount() const {
    return windowCount_;
}

}  // namespace emg
