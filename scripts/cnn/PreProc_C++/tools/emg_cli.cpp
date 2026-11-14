// emg_cli.cpp - outil PC pour les tests (pas pour la carte).
// emg_cli in.f32 n out_signal.f32 out_windows.f32 hp lp notch rect quant first chunk
#include "../src/emg_preprocessor.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc < 12) {
        std::cerr << "usage: emg_cli in n out_sig out_win hp lp notch rect quant first chunk\n";
        return 1;
    }
    const int n = std::atoi(argv[2]);
    std::vector<float> input(n * 8);
    std::ifstream inFile(argv[1], std::ios::binary);
    inFile.read(reinterpret_cast<char*>(input.data()), input.size() * sizeof(float));

    emg::PreprocessorConfig config;
    config.highpassEnabled  = std::atoi(argv[5]) != 0;
    config.lowpassEnabled   = std::atoi(argv[6]) != 0;
    config.notchEnabled     = std::atoi(argv[7]) != 0;
    config.rectifyEnabled   = std::atoi(argv[8]) != 0;
    config.quantizeEnabled  = std::atoi(argv[9]) != 0;
    config.firstWindowStart = std::atoi(argv[10]);
    const int chunk = std::atoi(argv[11]);

    // Deux préprocesseurs : un pour le signal complet, un pour les fenêtres.
    // (~34 Ko chacun : on les crée avec new sur PC, jamais sur la pile.)
    emg::EmgPreprocessor* forSignal  = new emg::EmgPreprocessor();
    emg::EmgPreprocessor* forWindows = new emg::EmgPreprocessor();
    if (forSignal->init(config) != emg::ErrorCode::Ok || forWindows->init(config) != emg::ErrorCode::Ok) {
        std::cerr << "config invalide\n";
        return 2;
    }

    // Signal complet (n, 8)
    std::vector<float> signal(n * 8);
    for (int i = 0; i < n; ++i) {
        for (int ch = 0; ch < 8; ++ch) {
            signal[i * 8 + ch] = forSignal->processValue(ch, input[i * 8 + ch]);
        }
    }
    std::ofstream sigFile(argv[3], std::ios::binary);
    sigFile.write(reinterpret_cast<const char*>(signal.data()), signal.size() * sizeof(float));

    // Fenêtres, en envoyant les données par blocs de "chunk" échantillons
    std::ofstream winFile(argv[4], std::ios::binary);
    for (int start = 0; start < n; start += chunk) {
        int count = chunk;
        if (start + count > n) {
            count = n - start;
        }
        for (int i = 0; i < count; ++i) {
            forWindows->pushSample(&input[(start + i) * 8]);
            if (forWindows->isWindowReady()) {
                winFile.write(reinterpret_cast<const char*>(forWindows->getWindow()),
                              8 * config.windowSize * sizeof(float));
                forWindows->clearWindowReady();
            }
        }
    }

    std::cout << "windows=" << forWindows->getWindowCount() << "\n";
    delete forSignal;
    delete forWindows;
    return 0;
}
