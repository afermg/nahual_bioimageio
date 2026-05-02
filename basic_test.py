"""Standalone smoke test for the bioimageio Nahual server.

Loads a known-good BIMZ model (default: ``affable-shark`` — a 2D nuclei
segmentation U-Net that publishes ONNX, TorchScript and pytorch_state_dict
weights) and runs a forward pass on a small synthetic input. No IPC.

Run from the repo root:
    nix develop --impure --command python basic_test.py
"""

import sys

# server.py reads sys.argv[1] at import time; inject a placeholder so
# importing it from this file doesn't crash.
if len(sys.argv) < 2:
    sys.argv.append("ipc:///tmp/bioimageio_basic_test.ipc")

import numpy  # noqa: E402

from server import setup  # noqa: E402


# affable-shark = 10.5281/zenodo.5764892, NucleiSegmentationBoundaryModel.
# Input axes: bcyx, 1 channel, 256x256.
MODEL = "affable-shark"


def main() -> None:
    processor, info = setup(source=MODEL)
    print(f"setup: {info}")
    assert "cuda" in info["device"], (
        f"Not on GPU! info['device']={info['device']!r}"
    )

    numpy.random.seed(0)
    data = numpy.random.random_sample((1, 1, 256, 256)).astype(numpy.float32)
    out = processor(data)

    arr = out.cpu().numpy() if hasattr(out, "cpu") else numpy.asarray(out)
    print(f"process: {type(arr).__name__} {arr.shape} dtype={arr.dtype}")
    # Sanity: the U-Net keeps the spatial dims; output channel may be 1 or 2.
    assert arr.shape[-2:] == (256, 256), (
        f"Unexpected spatial shape: got {arr.shape}, expected last two "
        f"dims to be (256, 256)."
    )


if __name__ == "__main__":
    main()
