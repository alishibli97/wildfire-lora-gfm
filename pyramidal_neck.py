import torch
import torch.nn as nn

# If you want exact TerraTorch compatibility, uncomment this:
# from terratorch.models.necks import Neck
#
# Otherwise you can define a lightweight placeholder:
class Neck(nn.Module):
    """Minimal Neck base class for standalone usage."""
    def __init__(self, channel_list=None):
        super().__init__()
        self.channel_list = channel_list


class LearnedInterpolateToPyramidal(Neck):
    """
    Use learned convolutions to transform the output of a non-pyramidal encoder
    into a pyramidal (multi-scale) feature set.

    Expected:
        - Exactly 4 input feature maps
        - channel_list: list of 4 ints, specifying input channel dims
    """

    def __init__(self, channel_list: list[int]):
        super().__init__(channel_list)

        if len(channel_list) != 4:
            raise ValueError(
                f"LearnedInterpolateToPyramidal requires 4 embeddings, got {len(channel_list)}"
            )

        # Highest-level / deepest features -> upsample 2x + reduce channels
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(channel_list[0], channel_list[0] // 2, 2, 2),
            nn.BatchNorm2d(channel_list[0] // 2),
            nn.GELU(),
            nn.ConvTranspose2d(channel_list[0] // 2, channel_list[0] // 4, 2, 2),
        )

        # Second level -> moderate upsample
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(channel_list[1], channel_list[1] // 2, 2, 2)
        )

        # Third level -> identity
        self.fpn3 = nn.Identity()

        # Fourth level -> downsample using maxpool
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Output channel sizes (needed by decoders)
        self.embedding_dim = [
            channel_list[0] // 4,  # fpn1 output channels
            channel_list[1] // 2,  # fpn2 output channels
            channel_list[2],       # identity
            channel_list[3],       # pooled
        ]

    def forward(self, features: list[torch.Tensor], **kwargs) -> list[torch.Tensor]:
        """
        Args:
            features: list of 4 tensors [B, C_i, H_i, W_i]

        Returns:
            list of 4 tensors scaled/reshaped into pyramid levels.
        """
        if len(features) != 4:
            raise ValueError(f"Expected 4 feature maps, got {len(features)}")

        scaled_inputs = [
            self.fpn1(features[0]),
            self.fpn2(features[1]),
            self.fpn3(features[2]),
            self.fpn4(features[3]),
        ]
        return scaled_inputs

    def process_channel_list(self, channel_list: list[int] = None) -> list[int]:
        """
        Returns the output channel sizes after the FPN transformations.
        """
        if channel_list is None:
            channel_list = self.channel_list
        return [
            channel_list[0] // 4,
            channel_list[1] // 2,
            channel_list[2],
            channel_list[3],
        ]
