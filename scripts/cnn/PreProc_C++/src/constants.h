// constants.h
// Tailles maximales, fixées à la compilation.
// Sur le microcontrôleur on n'utilise jamais new/malloc : toute la mémoire est réservée
// d'avance, donc il faut connaître les tailles maximales des tableaux.
#ifndef EMG_CONSTANTS_H
#define EMG_CONSTANTS_H

namespace emg {

const int MAX_CHANNELS = 8;     // nombre d'électrodes du bracelet Myo
const int MAX_WINDOW   = 512;   // taille max d'une fenêtre (on utilise 400)
const int MAX_ORDER    = 8;     // ordre max d'un filtre Butterworth (on utilise 1)

// Un Butterworth d'ordre N = (N+1)/2 biquads. HP + LP + notch :
const int MAX_SECTIONS = 2 * ((MAX_ORDER + 1) / 2) + 1;   // = 9

const double PI = 3.14159265358979323846;

}  // namespace emg

#endif

