import torch

from lada.models.basicvsrpp.basicvsrpp_gan import BasicVSRPlusPlusGan
from lada.utils import ImageTensor


class BasicvsrppMosaicRestorer:
    def __init__(self, model: BasicVSRPlusPlusGan, device: torch.device, fp16: bool):
        self.model = model
        self.device: torch.device = torch.device(device)
        self.dtype = torch.float16 if fp16 else torch.float32

    def prepare(self, video: list[ImageTensor]) -> tuple[torch.Tensor, int, tuple]:
        """Pre-process clip frames into a GPU tensor for inference.

        Can be called from a separate thread to overlap CPU preparation
        with ongoing GPU inference on the previous clip.
        """
        input_frame_count = len(video)
        input_frame_shape = video[0].shape
        inference_view = (
            torch.stack([x.permute(2, 0, 1) for x in video], dim=0)
            .to(device=self.device, non_blocking=True)
            .to(dtype=self.dtype)
            .div_(255.0)
            .unsqueeze(0)
        )
        return inference_view, input_frame_count, input_frame_shape

    def restore_prepared(
        self, prepared: tuple[torch.Tensor, int, tuple], max_frames: int = -1
    ) -> list[ImageTensor]:
        """Run GPU inference on a pre-processed tensor returned by prepare()."""
        inference_view, input_frame_count, input_frame_shape = prepared
        with torch.inference_mode():
            if max_frames > 0:
                result = []
                for i in range(0, inference_view.shape[1], max_frames):
                    output = self.model(inputs=inference_view[:, i : i + max_frames])
                    result.append(output)
                result = torch.cat(result, dim=1)
            else:
                result = self.model(inputs=inference_view)

            # (B, T, C, H, W) float in [0,1] to list of (H, W, C) uint8
            result = result.squeeze(0)[:input_frame_count]  # -> (T, C, H, W)
            result = (
                result.mul_(255.0)
                .round_()
                .clamp_(0, 255)
                .to(dtype=torch.uint8)
                .permute(0, 2, 3, 1)
            )  # (T, H, W, C)
            result = list(torch.unbind(result, 0))  # (T, H, W, C) to list of (H, W, C)
            output_frame_count = len(result)
            output_frame_shape = result[0].shape
            assert (
                input_frame_count == output_frame_count
                and input_frame_shape == output_frame_shape
            )

        return result

    def restore(
        self, video: list[ImageTensor], max_frames: int = -1
    ) -> list[ImageTensor]:
        prepared = self.prepare(video)
        return self.restore_prepared(prepared, max_frames)
