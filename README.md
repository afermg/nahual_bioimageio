# nahual_bioimageio

A single Nahual server that exposes the entire **BioImage Model Zoo** (BIMZ)
behind the standard `setup` / `process` IPC contract. You pass a model
identifier (DOI, Zenodo URL, nickname like `affable-shark`, or a local
`rdf.yaml` path) at `setup()` time, and the server loads it, picks the right
backend (PyTorch, TensorFlow, or ONNX), applies the RDF-declared
preprocessing / postprocessing, and gives you back numpy arrays.

## Why one server for the whole zoo?

`bioimageio.core` already speaks every backend BIMZ publishes. The wrap is
thin: it forwards `source` and `weight_format` to
`load_description` + `create_prediction_pipeline`, and translates between
numpy arrays and `xarray.DataArray`s using the axes the RDF declares.

## Run

```bash
# Default flavor: bioimageio.core + onnxruntime + torch + numpy + xarray.
nix run --impure github:afermg/nahual_bioimageio -- ipc:///tmp/bioimageio.ipc
```

Then from any Python process:

```python
from nahual.process import dispatch_setup_process
import numpy as np

setup, process = dispatch_setup_process("bioimageio", signature=("dict", "numpy"))

info = setup(
    {"source": "affable-shark"},                 # any BIMZ ID
    address="ipc:///tmp/bioimageio.ipc",
)
print(info)  # {'device': 'cuda:0', 'weight_format': 'onnx', 'input_axes': 'bcyx', ...}

img = np.random.random_sample((1, 1, 256, 256)).astype("float32")  # bcyx
out = process(img, address="ipc:///tmp/bioimageio.ipc")
print(out.shape)  # (1, 1, 256, 256) for affable-shark
```

## Flake variants

The same `server.py` is exposed through four `nix run` flavors that differ
only in the Python environment they ship. Pick the smallest one that works
for your model.

| App                  | Adds                | Use when…                          |
|----------------------|---------------------|------------------------------------|
| `.#default`          | nothing extra       | The model publishes ONNX or TorchScript weights (most of the zoo). |
| `.#with-stardist`    | `stardist`          | The RDF requires the StarDist Python package at load time. |
| `.#with-careamics`   | `careamics`         | The RDF requires CAREamics. |
| `.#with-monai`       | `monai`             | The RDF requires a MONAI architecture. |

```bash
nix run --impure github:afermg/nahual_bioimageio#with-stardist -- ipc:///tmp/bioimageio.ipc
```

## `setup()` parameters

| Param           | Type            | Default | Notes |
|-----------------|-----------------|---------|-------|
| `source`        | `str`           | —       | Required. Any identifier `bioimageio.core.load_description` accepts. |
| `weight_format` | `str \| None`   | `None`  | If `None`, tries `onnx` then `torchscript` then `pytorch_state_dict`. Errors out informatively if none of the preferred formats are published, listing what IS available. |
| `device`        | `int`           | `0`     | CUDA device index. Maps to `devices=["cuda:<device>"]`. |

## `process()` contract

Pass a numpy array whose dims match the RDF's declared input axes
(returned in the `input_axes` field of `setup()`'s response — e.g. `bcyx`).
The wrap reshapes it into an `xarray.DataArray`, runs the prediction
pipeline, and returns the first output as a plain numpy array. No `NCZYX`
padding here — different RDFs declare different axes; reshape on the
client side to match.

## Files

- `server.py` — Nahual server, multi-backend.
- `flake.nix` — four apps (`default` / `with-stardist` / `with-careamics`
  / `with-monai`) plus a dev shell.
- `nix/nahual.nix` — Nahual transport layer pin.
- `nix/bioimageio_core.nix`, `nix/bioimageio_spec.nix`,
  `nix/genericache.nix` — package these from PyPI; nixpkgs doesn't ship them.
- `basic_test.py` — standalone smoke test (loads `affable-shark`, runs
  forward pass, asserts cuda + sensible output shape).
