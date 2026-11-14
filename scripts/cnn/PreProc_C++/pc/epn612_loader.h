// epn612_loader.h
// PC ONLY (never on the board): reads one EMG-EPN-612 participant JSON file.
// C++ port of src/bmiemg/data/epn612/loader.py (load_user_trials).
#ifndef EMG_EPN612_LOADER_H
#define EMG_EPN612_LOADER_H

#include <string>
#include <vector>

namespace emg {

//------ Role: one recording (one gesture repetition) of one participant ------
struct EmgTrial {
    std::string userId;                      // userInfo.name
    std::string trialId;                     // e.g. "idx_12"
    std::string gesture;                     // gestureName ("" if absent)
    double samplingRateHz;                   // generalInfo.samplingFrequencyInHertz (200)
    std::vector<std::vector<float> > emg;    // emg[channel][sample], 8 channels

    bool hasGroundTruthIndex;                // groundTruthIndex present?
    int  groundTruthStart;                   // labelled interval [start, end]
    int  groundTruthEnd;
    int  gestureStart;                       // startPointforGestureExecution (-1 if absent)
    std::vector<int> groundTruth;            // per-sample 0/1 mask (empty if absent)

    int sampleCount() const { return emg.empty() ? 0 : static_cast<int>(emg[0].size()); }
};

//------ Role: load every trial of one section, sorted like Python (idx_1, idx_2, ..., idx_10) ------
// section = "trainingSamples" or "testingSamples".
// Throws std::runtime_error with a clear message if the file is missing or malformed.
std::vector<EmgTrial> loadUserTrials(const std::string& jsonPath,
                                     const std::string& section = "trainingSamples");

}  // namespace emg

#endif
