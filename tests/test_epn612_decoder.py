import numpy as np

from bmiemg.models.model_factories import SVMFactory
from bmiemg.pipeline import EMGSVMDecoder, OnlineEMGInference
from bmiemg.preprocessing import EMGProcessingConfig


def _decoder() -> EMGSVMDecoder:
    config = EMGProcessingConfig(
        sampling_rate_hz=200,
        window_size_samples=40,
        feature_names=("mav", "rms", "wl"),
        software_filter_enabled=False,
    )
    time = np.linspace(0, 2 * np.pi, 40, dtype=np.float32)
    rest = np.stack([np.sin(time + channel) for channel in range(8)])
    fist = rest * 10
    windows = np.stack([rest, rest * 1.1, fist, fist * 1.1])
    labels = np.array(["noGesture", "noGesture", "fist", "fist"])
    return EMGSVMDecoder(config, SVMFactory(C=1.0)).fit(windows, labels)


def test_decoder_predicts_and_round_trips(tmp_path):
    decoder = _decoder()
    window = np.ones((8, 40), dtype=np.float32)
    expected = decoder.predict_one(window)
    path = decoder.save(tmp_path / "decoder.joblib")

    loaded = EMGSVMDecoder.load(path)

    assert loaded.predict_one(window) == expected
    assert set(loaded.classes_) == {"fist", "noGesture"}


def test_online_buffer_emits_at_window_then_step():
    online = OnlineEMGInference(_decoder(), step_size_samples=20)

    first = online.add_samples(np.ones((8, 40), dtype=np.float32))
    second = online.add_samples(np.ones((8, 20), dtype=np.float32))

    assert len(first) == 1
    assert len(second) == 1
