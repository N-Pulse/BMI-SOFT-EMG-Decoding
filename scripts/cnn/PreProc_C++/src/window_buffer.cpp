// window_buffer.cpp
#include "window_buffer.h"

namespace emg {

WindowBuffer::WindowBuffer() {
    configure(8, 400, 200, 0);
}

void WindowBuffer::configure(int channels, int windowSize, int windowStep, int firstWindowStart) {
    channels_ = channels;
    windowSize_ = windowSize;
    windowStep_ = windowStep;
    firstWindowStart_ = firstWindowStart;
    reset();
}

void WindowBuffer::reset() {
    writeIndex_ = 0;
    samplesReceived_ = 0;
    // La 1ère fenêtre couvre [firstWindowStart, firstWindowStart + windowSize) :
    // elle est complète quand on a reçu firstWindowStart + windowSize échantillons.
    nextWindowEnd_ = firstWindowStart_ + windowSize_;
}

bool WindowBuffer::push(const float frame[]) {
    // 1. Écrire l'échantillon de chaque canal à la position courante.
    for (int channel = 0; channel < channels_; ++channel) {
        ring_[channel][writeIndex_] = frame[channel];
    }

    // 2. Avancer la position ; revenir à 0 au bout (c'est ce qui rend le buffer "circulaire").
    writeIndex_++;
    if (writeIndex_ == windowSize_) {
        writeIndex_ = 0;
    }
    samplesReceived_++;

    // 3. Une fenêtre est-elle complète ?
    if (samplesReceived_ == nextWindowEnd_) {
        copyWindow();
        nextWindowEnd_ += windowStep_;   // la suivante sera complète "step" échantillons plus tard
        return true;
    }
    return false;
}

void WindowBuffer::copyWindow() {
    // Après l'écriture, writeIndex_ pointe sur le plus VIEIL échantillon du buffer.
    // On lit donc à partir de là, en faisant le tour.
    for (int channel = 0; channel < channels_; ++channel) {
        int readIndex = writeIndex_;
        for (int i = 0; i < windowSize_; ++i) {
            window_[channel * windowSize_ + i] = ring_[channel][readIndex];
            readIndex++;
            if (readIndex == windowSize_) {
                readIndex = 0;
            }
        }
    }
}

const float* WindowBuffer::getWindow() const {
    return window_;
}

}  // namespace emg
