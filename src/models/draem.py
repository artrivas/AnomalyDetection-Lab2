"""DRAEM model wrapper."""

from __future__ import annotations

import torch
from torch import nn

from src.models.discriminative_subnetwork import DiscriminativeSubNetwork
from src.models.reconstructive_subnetwork import ReconstructiveSubNetwork


class DRAEM(nn.Module):
    """DRAEM reconstruction and anomaly segmentation model."""

    def __init__(
        self,
        reconstructive: ReconstructiveSubNetwork | None = None,
        discriminative: DiscriminativeSubNetwork | None = None,
    ) -> None:
        super().__init__()
        self.reconstructive = reconstructive or ReconstructiveSubNetwork(
            in_channels=3,
            out_channels=3,
        )
        self.discriminative = discriminative or DiscriminativeSubNetwork(
            in_channels=6,
            out_channels=1,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        reconstruction = self.reconstructive(x)
        concatenated = torch.cat([x, reconstruction], dim=1)
        logits = self.discriminative(concatenated)
        anomaly_map = torch.sigmoid(logits)
        return {
            "reconstruction": reconstruction,
            "logits": logits,
            "anomaly_map": anomaly_map,
        }


def _smoke_test() -> None:
    model = DRAEM()
    model.eval()

    x = torch.rand(2, 3, 256, 256)
    with torch.no_grad():
        outputs = model(x)

    assert outputs["reconstruction"].shape == (2, 3, 256, 256)
    assert outputs["logits"].shape == (2, 1, 256, 256)
    assert outputs["anomaly_map"].shape == (2, 1, 256, 256)
    assert outputs["reconstruction"].min() >= 0.0
    assert outputs["reconstruction"].max() <= 1.0
    print("DRAEM smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
