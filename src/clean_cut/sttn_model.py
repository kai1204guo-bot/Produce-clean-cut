"""STTN inference network.

Adapted from researchmm/STTN (commit f39f62c) under the MIT License.
Only the generator needed for inference is retained and modernized.
"""

from __future__ import annotations

import math
from typing import Any


def build_sttn_generator() -> Any:
    try:
        import torch
        from torch import nn
        from torch.nn import functional as functional
    except ImportError as exc:
        from clean_cut.errors import CleanCutError

        raise CleanCutError("STTN后端需要PyTorch，请安装sttn可选依赖。") from exc

    class Deconv(nn.Module):
        def __init__(self, input_channel: int, output_channel: int) -> None:
            super().__init__()
            self.conv = nn.Conv2d(input_channel, output_channel, 3, padding=1)

        def forward(self, value: Any) -> Any:
            value = functional.interpolate(
                value,
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            )
            return self.conv(value)

    class Attention(nn.Module):
        def forward(self, query: Any, key: Any, value: Any, mask: Any) -> Any:
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
            scores.masked_fill_(mask, -1e9)
            return torch.matmul(functional.softmax(scores, dim=-1), value)

    class MultiHeadedAttention(nn.Module):
        def __init__(self, patch_sizes: tuple[tuple[int, int], ...], channels: int) -> None:
            super().__init__()
            self.patch_sizes = patch_sizes
            self.query_embedding = nn.Conv2d(channels, channels, 1)
            self.value_embedding = nn.Conv2d(channels, channels, 1)
            self.key_embedding = nn.Conv2d(channels, channels, 1)
            self.output_linear = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.attention = Attention()

        def forward(self, value: Any, mask: Any, batch: int, channels: int) -> Any:
            batch_time, _, height, width = value.size()
            time = batch_time // batch
            head_channels = channels // len(self.patch_sizes)
            outputs = []
            queries = torch.chunk(self.query_embedding(value), len(self.patch_sizes), dim=1)
            keys = torch.chunk(self.key_embedding(value), len(self.patch_sizes), dim=1)
            values = torch.chunk(self.value_embedding(value), len(self.patch_sizes), dim=1)

            def reshape_patches(
                tensor: Any,
                output_height: int,
                output_width: int,
                item_patch_height: int,
                item_patch_width: int,
            ) -> Any:
                tensor = tensor.view(
                    batch,
                    time,
                    head_channels,
                    output_height,
                    item_patch_height,
                    output_width,
                    item_patch_width,
                )
                tensor = tensor.permute(0, 1, 3, 5, 2, 4, 6)
                return tensor.contiguous().view(
                    batch,
                    time * output_height * output_width,
                    head_channels * item_patch_height * item_patch_width,
                )

            for (patch_width, patch_height), query, key, item in zip(
                self.patch_sizes,
                queries,
                keys,
                values,
                strict=True,
            ):
                out_width = width // patch_width
                out_height = height // patch_height
                attention_mask = mask.view(
                    batch,
                    time,
                    1,
                    out_height,
                    patch_height,
                    out_width,
                    patch_width,
                )
                attention_mask = attention_mask.permute(0, 1, 3, 5, 2, 4, 6)
                attention_mask = attention_mask.contiguous().view(
                    batch,
                    time * out_height * out_width,
                    patch_height * patch_width,
                )
                attention_mask = (attention_mask.mean(-1) > 0.5).unsqueeze(1)
                attention_mask = attention_mask.repeat(
                    1,
                    time * out_height * out_width,
                    1,
                )

                attended = self.attention(
                    reshape_patches(query, out_height, out_width, patch_height, patch_width),
                    reshape_patches(key, out_height, out_width, patch_height, patch_width),
                    reshape_patches(item, out_height, out_width, patch_height, patch_width),
                    attention_mask,
                )
                attended = attended.view(
                    batch,
                    time,
                    out_height,
                    out_width,
                    head_channels,
                    patch_height,
                    patch_width,
                )
                attended = attended.permute(0, 1, 4, 2, 5, 3, 6)
                outputs.append(attended.contiguous().view(batch_time, head_channels, height, width))
            return self.output_linear(torch.cat(outputs, dim=1))

    class FeedForward(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=2, dilation=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )

        def forward(self, value: Any) -> Any:
            return self.conv(value)

    class TransformerBlock(nn.Module):
        def __init__(self, patch_sizes: tuple[tuple[int, int], ...], channels: int) -> None:
            super().__init__()
            self.attention = MultiHeadedAttention(patch_sizes, channels)
            self.feed_forward = FeedForward(channels)

        def forward(self, state: dict[str, Any]) -> dict[str, Any]:
            value, mask = state["x"], state["m"]
            value = value + self.attention(value, mask, state["b"], state["c"])
            value = value + self.feed_forward(value)
            return {"x": value, "m": mask, "b": state["b"], "c": state["c"]}

    class InpaintGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            channels = 256
            patch_sizes = ((108, 60), (36, 20), (18, 10), (9, 5))
            self.transformer = nn.Sequential(
                *(TransformerBlock(patch_sizes, channels) for _ in range(8))
            )
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, channels, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.decoder = nn.Sequential(
                Deconv(channels, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                Deconv(64, 64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 3, 3, padding=1),
            )

        def infer(self, features: Any, masks: Any) -> Any:
            time, channels, _, _ = features.size()
            masks = functional.interpolate(masks.view(time, 1, 240, 432), scale_factor=0.25)
            return self.transformer(
                {"x": features, "m": masks, "b": 1, "c": channels}
            )["x"]

    return InpaintGenerator()
