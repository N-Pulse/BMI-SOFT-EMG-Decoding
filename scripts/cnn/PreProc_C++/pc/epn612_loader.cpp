// epn612_loader.cpp
#include "epn612_loader.h"
#include "../third_party/json.hpp"

#include <algorithm>
#include <cctype>
#include <climits>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace emg {

namespace {

using Json = nlohmann::json;

//------ Role: throw a readable error that names the file and the trial ------
void fail(const std::string& message, const std::string& path, const std::string& trialId = "") {
    throw std::runtime_error(message + " at " + path + (trialId.empty() ? "" : ":" + trialId));
}

//------ Role: get a sub-object, or fail if it is missing / not an object ------
const Json& requireObject(const Json& parent, const std::string& key,
                          const std::string& path, const std::string& trialId = "") {
    if (!parent.contains(key) || !parent[key].is_object()) {
        fail("Missing object '" + key + "'", path, trialId);
    }
    return parent[key];
}

//------ Role: sort key "idx_12" -> 12 (same as Python _numeric_key) ------
long numericKey(const std::string& trialId) {
    const std::size_t underscore = trialId.rfind('_');
    const std::string suffix = (underscore == std::string::npos) ? trialId : trialId.substr(underscore + 1);
    if (suffix.empty()) return LONG_MAX;
    for (std::size_t i = 0; i < suffix.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(suffix[i]))) return LONG_MAX;
    }
    return std::stol(suffix);
}

bool trialOrder(const std::pair<long, std::string>& a, const std::pair<long, std::string>& b) {
    if (a.first != b.first) return a.first < b.first;
    return a.second < b.second;
}

//------ Role: turn one JSON trial into an EmgTrial (same checks as Python _parse_trial) ------
EmgTrial parseTrial(const Json& sample, double samplingRate, const std::string& userId,
                    const std::string& trialId, const std::string& path) {
    EmgTrial trial;
    trial.userId = userId;
    trial.trialId = trialId;
    trial.samplingRateHz = samplingRate;

    // --- EMG: 8 channels "ch1".."ch8", all the same length ---
    const Json& emgObject = requireObject(sample, "emg", path, trialId);
    trial.emg.resize(8);
    for (int ch = 0; ch < 8; ++ch) {
        const std::string name = "ch" + std::to_string(ch + 1);
        if (!emgObject.contains(name) || !emgObject[name].is_array()) {
            fail("Missing or invalid " + name, path, trialId);
        }
        const Json& values = emgObject[name];
        trial.emg[ch].reserve(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (!values[i].is_number()) fail(name + " contains a non-number", path, trialId);
            trial.emg[ch].push_back(values[i].get<float>());
        }
        if (ch > 0 && trial.emg[ch].size() != trial.emg[0].size()) {
            fail("Unequal EMG channel lengths", path, trialId);
        }
    }
    if (trial.emg[0].empty()) fail("Empty EMG", path, trialId);

    // --- Optional annotations ---
    trial.gesture = "";
    if (sample.contains("gestureName") && sample["gestureName"].is_string()) {
        trial.gesture = sample["gestureName"].get<std::string>();
    }

    trial.hasGroundTruthIndex = false;
    trial.groundTruthStart = trial.groundTruthEnd = -1;
    if (sample.contains("groundTruthIndex") && !sample["groundTruthIndex"].is_null()) {
        const Json& gti = sample["groundTruthIndex"];
        if (!gti.is_array() || gti.size() != 2 || !gti[0].is_number() || !gti[1].is_number()) {
            fail("Invalid groundTruthIndex", path, trialId);
        }
        trial.groundTruthStart = gti[0].get<int>();
        trial.groundTruthEnd = gti[1].get<int>();
        if (trial.groundTruthStart < 0 || trial.groundTruthEnd < trial.groundTruthStart) {
            fail("Invalid groundTruthIndex", path, trialId);
        }
        trial.hasGroundTruthIndex = true;
    }

    trial.gestureStart = -1;
    if (sample.contains("startPointforGestureExecution") &&
        sample["startPointforGestureExecution"].is_number_integer()) {
        trial.gestureStart = sample["startPointforGestureExecution"].get<int>();
    }

    if (sample.contains("groundTruth") && sample["groundTruth"].is_array()) {
        const Json& gt = sample["groundTruth"];
        trial.groundTruth.reserve(gt.size());
        for (std::size_t i = 0; i < gt.size(); ++i) trial.groundTruth.push_back(gt[i].get<int>());
    }
    return trial;
}

}  // namespace

std::vector<EmgTrial> loadUserTrials(const std::string& jsonPath, const std::string& section) {
    if (section != "trainingSamples" && section != "testingSamples") {
        throw std::runtime_error("Invalid section '" + section + "'");
    }

    // 1. Read and parse the whole file.
    std::ifstream file(jsonPath.c_str());
    if (!file) throw std::runtime_error("File not found: " + jsonPath);
    Json document;
    try {
        file >> document;
    } catch (const std::exception& error) {
        throw std::runtime_error("Invalid JSON in " + jsonPath + ": " + error.what());
    }
    if (!document.is_object()) fail("Expected a JSON object", jsonPath);

    // 2. File-level info.
    const Json& general = requireObject(document, "generalInfo", jsonPath);
    const Json& user = requireObject(document, "userInfo", jsonPath);
    const Json& samples = requireObject(document, section, jsonPath);

    if (!general.contains("samplingFrequencyInHertz") || !general["samplingFrequencyInHertz"].is_number()) {
        fail("Invalid samplingFrequencyInHertz", jsonPath);
    }
    const double samplingRate = general["samplingFrequencyInHertz"].get<double>();
    if (!(samplingRate > 0.0)) fail("Invalid samplingFrequencyInHertz", jsonPath);

    if (!user.contains("name") || !user["name"].is_string()) fail("Invalid userInfo.name", jsonPath);
    const std::string userId = user["name"].get<std::string>();

    // 3. Trials, sorted numerically: idx_1, idx_2, ..., idx_10 (not idx_1, idx_10, idx_2).
    std::vector<std::pair<long, std::string> > ids;
    for (Json::const_iterator it = samples.begin(); it != samples.end(); ++it) {
        ids.push_back(std::make_pair(numericKey(it.key()), it.key()));
    }
    std::sort(ids.begin(), ids.end(), trialOrder);

    std::vector<EmgTrial> trials;
    trials.reserve(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const std::string& trialId = ids[i].second;
        const Json& sample = samples[trialId];
        if (!sample.is_object()) fail("Invalid trial entry", jsonPath, trialId);
        trials.push_back(parseTrial(sample, samplingRate, userId, trialId, jsonPath));
    }
    return trials;
}

}  // namespace emg
