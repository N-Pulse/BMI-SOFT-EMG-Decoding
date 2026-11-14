// window_buffer.h
// Garde les derniers échantillons de chaque canal dans un buffer circulaire
// et sort une fenêtre (canaux x taille) tous les "step" échantillons.
#ifndef EMG_WINDOW_BUFFER_H
#define EMG_WINDOW_BUFFER_H

#include "constants.h"

namespace emg {

class WindowBuffer {
public:
    WindowBuffer();

    // firstWindowStart : indice de l'échantillon où commence la 1ère fenêtre
    // (permet de reproduire le "first" du code Python).
    void configure(int channels, int windowSize, int windowStep, int firstWindowStart);

    // Vide le buffer et remet les compteurs à zéro.
    void reset();

    // Ajoute un échantillon par canal. Renvoie true si une nouvelle fenêtre est complète.
    bool push(const float frame[]);

    // Dernière fenêtre complète, rangée canal par canal :
    // [canal 0 : windowSize valeurs][canal 1 : windowSize valeurs]...
    const float* getWindow() const;

private:
    // Recopie le buffer circulaire dans window_, dans l'ordre chronologique.
    void copyWindow();

    int channels_;
    int windowSize_;
    int windowStep_;
    int firstWindowStart_;

    float ring_[MAX_CHANNELS][MAX_WINDOW];      // buffer circulaire
    float window_[MAX_CHANNELS * MAX_WINDOW];   // fenêtre remise à plat

    int  writeIndex_;          // prochaine case à écrire dans ring_
    long samplesReceived_;     // nombre total d'échantillons reçus
    long nextWindowEnd_;       // samplesReceived_ auquel la prochaine fenêtre sera complète
};

}  // namespace emg

#endif
