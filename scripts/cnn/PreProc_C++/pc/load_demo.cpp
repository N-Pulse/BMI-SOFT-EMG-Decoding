// load_demo.cpp - PC only: load one EPN-612 participant and print what was read.
// Usage: ./load_demo path/to/userX.json [trainingSamples|testingSamples]
#include "epn612_loader.h"

#include <exception>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "usage: load_demo path/to/userX.json [trainingSamples|testingSamples]\n";
        return 1;
    }
    const std::string section = (argc >= 3) ? argv[2] : "trainingSamples";

    try {
        const std::vector<emg::EmgTrial> trials = emg::loadUserTrials(argv[1], section);
        std::cout << trials.size() << " trials loaded from " << argv[1] << " (" << section << ")\n";
        for (std::size_t i = 0; i < trials.size() && i < 5; ++i) {
            const emg::EmgTrial& t = trials[i];
            std::cout << "  " << t.trialId << "  gesture=" << (t.gesture.empty() ? "-" : t.gesture)
                      << "  shape=(8, " << t.sampleCount() << ")"
                      << "  fs=" << t.samplingRateHz << " Hz"
                      << "  ch1[0..3]=" << t.emg[0][0] << " " << t.emg[0][1] << " " << t.emg[0][2];
            if (t.hasGroundTruthIndex) std::cout << "  gt=[" << t.groundTruthStart << ", " << t.groundTruthEnd << "]";
            std::cout << "\n";
        }
        if (trials.size() > 5) std::cout << "  ...\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 2;
    }
    return 0;
}
