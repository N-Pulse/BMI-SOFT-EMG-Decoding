# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class SNR:
    per_channel: dict
    mean: float

    @property
    def summary(self) -> None:
        for key, value in self.per_channel.items():
            print(f"SNR for channel {key}: {round(value, 2)}dB")
        print(f"Mean SNR: {round(self.mean, 2)}dB")
