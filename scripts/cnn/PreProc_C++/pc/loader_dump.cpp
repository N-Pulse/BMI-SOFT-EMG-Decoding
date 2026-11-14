// loader_dump.cpp - PC only, for tests: load a participant JSON with the C++ loader
// and write back what was understood, as JSON, so Python can compare it.
// Usage: ./loader_dump input.json output.json [trainingSamples|testingSamples]
#include "epn612_loader.h"
#include "../third_party/json.hpp"

#include <exception>
#include <fstream>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "usage: loader_dump input.json output.json [section]\n";
        return 1;
    }
    const std::string section = (argc >= 4) ? argv[3] : "trainingSamples";

    try {
        const std::vector<emg::EmgTrial> trials = emg::loadUserTrials(argv[1], section);

        nlohmann::json out = nlohmann::json::array();
        for (std::size_t i = 0; i < trials.size(); ++i) {
            const emg::EmgTrial& t = trials[i];
            nlohmann::json item;
            item["userId"] = t.userId;
            item["trialId"] = t.trialId;
            item["gesture"] = t.gesture;
            item["samplingRateHz"] = t.samplingRateHz;
            item["emg"] = t.emg;
            item["groundTruthIndex"] = t.hasGroundTruthIndex
                ? nlohmann::json::array({t.groundTruthStart, t.groundTruthEnd})
                : nlohmann::json(nullptr);
            item["gestureStart"] = t.gestureStart;
            item["groundTruth"] = t.groundTruth;
            out.push_back(item);
        }
        std::ofstream file(argv[2]);
        file << out;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 2;
    }
    return 0;
}
