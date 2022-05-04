from typing import Optional
import torch
import torchaudio

class CustomTransform(torch.nn.Module):
    r"""
    """

    def __init__(self, sample_rate: int = 16_000, speed: Optional[float] = 0.8, lowpass: Optional[int] = 300) -> None:
        super(CustomTransform, self).__init__()
        self.sample_rate = sample_rate
        self.speed = speed
        self.lowpass = lowpass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """

        # Define effects
        effects = [
            ["lowpass", "-1", f"{self.lowpass}"], # apply single-pole lowpass filter
            ["speed", f"{self.speed}"],  # reduce the speed
                                # This only changes sample rate, so it is necessary to
                                # add `rate` effect with original sample rate after this.
            ["rate", f"{self.sample_rate}"],
            ["reverb", "-w"],  # Reverbration gives some dramatic feeling
        ]

        return torchaudio.sox_effects.apply_effects_tensor(x, self.sample_rate, effects)[0]
        