"""Nahual server for the BioImage Model Zoo (BIMZ).

A single Nahual server that can load ANY model from the BioImage Model Zoo
by passing its identifier (DOI, Zenodo URL, nickname like "affable-shark",
or a local rdf.yaml path) at ``setup()`` time.

The heavy lifting is done by ``bioimageio.core``: it parses the RDF, picks
the right backend (PyTorch / TensorFlow / ONNX), runs preprocessing
(``zero_mean_unit_variance``, ``scale_range``) and postprocessing
(``sigmoid``, ``binarize``) declared in the RDF, and returns an xarray
``DataArray`` (wrapped in a Sample) with the same axes the model declares.

Run with:
    nix run --impure . -- ipc:///tmp/bioimageio.ipc
or:
    python server.py ipc:///tmp/bioimageio.ipc
"""

import sys
from functools import partial
from typing import Callable

import numpy
import pynng
import trio
from nahual.server import responder

# server.py captures argv[1] at import time; basic_test.py injects a
# placeholder before importing this module.
address = sys.argv[1]


# Order in which we try weight formats when the caller doesn't specify one.
# ONNX first because it's lightweight, broadly compatible, and the GPU
# execution provider in nixpkgs' onnxruntime is built with cudaSupport.
# TorchScript next: still GPU via torch. Pytorch state-dict last because it
# requires the model architecture to be importable in the current env.
_WEIGHT_FORMAT_PRIORITY = (
    "onnx",
    "torchscript",
    "pytorch_state_dict",
    "tensorflow_saved_model_bundle",
    "keras_hdf5",
)


def _list_available_weight_formats(desc) -> list[str]:
    """Return the names of weight formats actually published by an RDF."""
    weights = getattr(desc, "weights", None)
    if weights is None:
        return []
    available = []
    for fmt in (
        "pytorch_state_dict",
        "torchscript",
        "onnx",
        "tensorflow_saved_model_bundle",
        "keras_hdf5",
        "tensorflow_js",
    ):
        if getattr(weights, fmt, None) is not None:
            available.append(fmt)
    return available


def setup(
    source: str,
    weight_format: str | None = None,
    device: int = 0,
) -> tuple[Callable, dict]:
    """Load a BIMZ model and return (processor_partial, info_dict).

    Parameters
    ----------
    source : str
        Anything ``bioimageio.core.load_description`` accepts: a model
        nickname (``"affable-shark"``), a DOI, a Zenodo URL, or a local
        ``rdf.yaml`` path.
    weight_format : str | None
        One of ``onnx``, ``torchscript``, ``pytorch_state_dict``,
        ``tensorflow_saved_model_bundle``. If None, we try ``onnx`` then
        ``torchscript``; setup raises informatively if neither is available
        AND nothing else in our priority list works either.
    device : int
        CUDA device index. Used for the prediction-pipeline's ``devices``
        argument as ``cuda:<device>``.
    """
    # Imports are inside setup so importing this module (e.g. for
    # basic_test.py) doesn't pay the cost.
    from bioimageio.core import create_prediction_pipeline, load_description

    desc = load_description(source)

    available = _list_available_weight_formats(desc)
    if not available:
        raise RuntimeError(
            f"RDF for {source!r} declares no weight entries — cannot load."
        )

    if weight_format is None:
        candidates = [w for w in _WEIGHT_FORMAT_PRIORITY if w in available]
        if not candidates:
            raise RuntimeError(
                f"None of the preferred weight formats "
                f"{_WEIGHT_FORMAT_PRIORITY!r} are available for {source!r}. "
                f"RDF publishes: {available!r}. Pass `weight_format=` "
                f"explicitly to use one of them."
            )
    else:
        if weight_format not in available:
            raise RuntimeError(
                f"weight_format={weight_format!r} not published by {source!r}. "
                f"Available: {available!r}."
            )
        candidates = [weight_format]

    devices = [f"cuda:{int(device)}"]

    pipe = None
    last_err = None
    chosen = None
    for cand in candidates:
        try:
            pipe = create_prediction_pipeline(
                desc, weight_format=cand, devices=devices
            )
            chosen = cand
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue

    if pipe is None:
        raise RuntimeError(
            f"Failed to load any of {candidates!r} for {source!r}. "
            f"Last error: {last_err!r}. Available formats: {available!r}."
        )

    # Compute a string representation of input axes (e.g. "bcyx") that works
    # for both v0_4 (axes is a string) and v0_5 (axes is a list of Axis).
    from bioimageio.core.digest_spec import get_axes_infos
    input_axes = "".join(str(ai.id) for ai in get_axes_infos(desc.inputs[0]))
    info = {
        "device": devices[0],
        "model_id": str(getattr(desc, "id", "") or ""),
        "name": str(getattr(desc, "name", "") or ""),
        "input_axes": input_axes,
        "weight_format": chosen,
        "available_weight_formats": available,
        "source": source,
    }

    processor = partial(_run, pipe=pipe, desc=desc)
    return processor, info


def _run(pixels: numpy.ndarray, pipe, desc) -> numpy.ndarray:
    """Run a single forward pass on a numpy array.

    The array's leading dims must match the RDF's declared input axes
    (``desc.inputs[0].axes``) — the BIMZ contract is per-model. For a
    typical 2D segmentation model that's ``bcyx``: ``(N, C, H, W)``.

    We wrap the array in an ``xarray.DataArray`` with the right dims, hand
    it to ``bioimageio.core.predict``, and convert the first output back to
    numpy.
    """
    import xarray as xr
    from bioimageio.core import predict
    from bioimageio.core.digest_spec import get_axes_infos, get_member_id

    axes = tuple(str(ai.id) for ai in get_axes_infos(desc.inputs[0]))
    if pixels.ndim != len(axes):
        raise ValueError(
            f"Input array has ndim={pixels.ndim} but the RDF declares "
            f"axes={axes!r} (ndim={len(axes)}). Reshape to match before "
            f"sending."
        )
    arr = xr.DataArray(pixels, dims=axes)

    sample = predict(model=pipe, inputs=arr)

    # `sample` is a bioimageio.core Sample with a `.members` dict. Pluck the
    # first declared output (handles both v0_4 .name and v0_5 .id RDFs) and
    # return as numpy.
    out_id = get_member_id(desc.outputs[0])
    tensor = sample.members[out_id]
    # Tensor wraps an xarray.DataArray under .data
    data = getattr(tensor, "data", tensor)
    if hasattr(data, "to_numpy"):
        return data.to_numpy()
    return numpy.asarray(data)


async def main():
    with pynng.Rep0(listen=address, recv_timeout=300) as sock:
        print(f"bioimageio server listening on {address}", flush=True)
        async with trio.open_nursery() as nursery:
            nursery.start_soon(partial(responder, setup=setup), sock)


if __name__ == "__main__":
    try:
        trio.run(main)
    except KeyboardInterrupt:
        pass
