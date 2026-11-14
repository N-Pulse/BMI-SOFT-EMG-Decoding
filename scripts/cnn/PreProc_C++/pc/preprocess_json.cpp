// preprocess_json.cpp - PC ONLY: participant JSON file  ->  preprocessed windows.
//
//   userX.json --[epn612_loader]--> trials --[EmgPreprocessor, same code as the board]--> windows
//
// Usage: ./preprocess_json input.json output_prefix [section] [input_bits]
//   section    : trainingSamples (default) or testingSamples
//   input_bits : resolution of the raw values in the file (default 8 = EPN-612 int8).
//                Values are normalized like Python: value / 2^(input_bits - 1).
// Writes:
//   output_prefix_windows.npy : float32 array (n_windows, channels, window_size), open with np.load
//   output_prefix_index.csv   : one line per window: which trial, gesture, start sample, labelled interval
#include "epn612_loader.h"
#include "emg_preprocessor.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

//------ Role: one output window + where it comes from ------
struct WindowInfo {
    std::string trialId;
    std::string gesture;
    long startSample;
    int groundTruthStart;
    int groundTruthEnd;
};

//------ Role: write a float32 array in NumPy's .npy format (readable with np.load) ------
// Format: magic "\x93NUMPY", version 1.0, header length (2 bytes), text header, raw float32 data.
void writeNpy(const std::string& path, const std::vector<float>& data,
              long n0, long n1, long n2) {
    std::ostringstream header;
    header << "{'descr': '<f4', 'fortran_order': False, 'shape': ("
           << n0 << ", " << n1 << ", " << n2 << "), }";
    std::string text = header.str();
    // Total header (10 fixed bytes + text + '\n') must be a multiple of 64 bytes.
    const std::size_t total = 10 + text.size() + 1;
    text += std::string((64 - total % 64) % 64, ' ');
    text += '\n';

    std::ofstream file(path.c_str(), std::ios::binary);
    if (!file) throw std::runtime_error("Cannot write " + path);
    const char magic[] = {'\x93', 'N', 'U', 'M', 'P', 'Y', 1, 0};
    file.write(magic, 8);
    const std::uint16_t headerLength = static_cast<std::uint16_t>(text.size());
    const char lengthBytes[] = {static_cast<char>(headerLength & 0xFF),
                                static_cast<char>(headerLength >> 8)};   // little-endian
    file.write(lengthBytes, 2);
    file.write(text.data(), static_cast<std::streamsize>(text.size()));
    file.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size() * sizeof(float)));
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "usage: preprocess_json input.json output_prefix [section] [input_bits]\n";
        return 1;
    }
    const std::string inputPath = argv[1];
    const std::string outputPrefix = argv[2];
    const std::string section = (argc >= 4) ? argv[3] : "trainingSamples";
    const int inputBits = (argc >= 5) ? std::atoi(argv[4]) : 8;

    // ------------------------------------------------------------------
    // Preprocessing settings: change them here (defaults = train.py).
    // ------------------------------------------------------------------
    emg::PreprocessorConfig config;
    // config.notchEnabled = false;
    // config.windowSize = 400;
    // config.windowStep = 200;

    try {
        // 1. Load the file.
        const std::vector<emg::EmgTrial> trials = emg::loadUserTrials(inputPath, section);
        std::cout << trials.size() << " trials loaded from " << inputPath << "\n";

        // 2. Create the preprocessor (~34 KB: on the heap, not on the stack).
        std::unique_ptr<emg::EmgPreprocessor> pre(new emg::EmgPreprocessor());
        if (pre->init(config) != emg::ErrorCode::Ok) throw std::runtime_error("Invalid preprocessing config");

        const float scale = static_cast<float>(1.0 / std::ldexp(1.0, inputBits - 1));
        const int channels = config.channels;
        const int windowSize = config.windowSize;

        std::vector<float> allWindows;
        std::vector<WindowInfo> index;

        // 3. Process each trial independently (filters start from zero, like Python).
        for (std::size_t t = 0; t < trials.size(); ++t) {
            const emg::EmgTrial& trial = trials[t];

            if (std::fabs(trial.samplingRateHz - config.samplingRateHz) > 1e-6) {
                std::ostringstream message;
                message << trial.trialId << " is sampled at " << trial.samplingRateHz
                        << " Hz but the preprocessing expects " << config.samplingRateHz
                        << " Hz (resampling is not implemented)";
                throw std::runtime_error(message.str());
            }

            pre->reset();
            const int n = trial.sampleCount();
            for (int i = 0; i < n; ++i) {
                float frame[emg::MAX_CHANNELS];
                for (int ch = 0; ch < channels; ++ch) {
                    frame[ch] = trial.emg[ch][i] * scale;          // same as Python: value / 2^(bits-1)
                }
                pre->pushSample(frame);

                if (pre->isWindowReady()) {
                    const float* window = pre->getWindow();
                    allWindows.insert(allWindows.end(), window, window + channels * windowSize);

                    WindowInfo info;
                    info.trialId = trial.trialId;
                    info.gesture = trial.gesture;
                    info.startSample = i + 1 - windowSize;
                    info.groundTruthStart = trial.groundTruthStart;
                    info.groundTruthEnd = trial.groundTruthEnd;
                    index.push_back(info);

                    pre->clearWindowReady();
                }
            }
        }

        // 4. Save.
        const std::string npyPath = outputPrefix + "_windows.npy";
        const std::string csvPath = outputPrefix + "_index.csv";
        writeNpy(npyPath, allWindows, static_cast<long>(index.size()), channels, windowSize);

        std::ofstream csv(csvPath.c_str());
        csv << "window,trial_id,gesture,start_sample,ground_truth_start,ground_truth_end\n";
        for (std::size_t w = 0; w < index.size(); ++w) {
            csv << w << "," << index[w].trialId << "," << index[w].gesture << ","
                << index[w].startSample << "," << index[w].groundTruthStart << ","
                << index[w].groundTruthEnd << "\n";
        }

        std::cout << index.size() << " windows of shape (" << channels << ", " << windowSize
                  << ") written to " << npyPath << " and " << csvPath << "\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 2;
    }
    return 0;
}
