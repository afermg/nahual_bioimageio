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

## Validated models from the BioImage Model Zoo

These BIMZ entries have been verified end-to-end on GPU through the
`server.setup()` + `server._run()` code path: each loads from a fresh
`bioimageio.core` cache, lands on `cuda:0`, and produces a sensible-shaped
forward pass on synthetic input. Picked from the 44-model BIMZ collection
JSON by (a) high download count, (b) `onnx` or `torchscript` weights
published in the RDF (the `default` flake variant has no model-architecture
dependencies), and (c) coverage spread across nuclei/cell/EM segmentation,
super-resolution, label-free, and classification.

```bash
nix run --impure github:afermg/nahual_bioimageio -- ipc:///tmp/bioimageio.ipc
```

Click any nickname to view the model card on `bioimage.io`.

| Nickname | Name | Task | Input axes (shape) | Weight format | Output shape | App variant |
|---|---|---|---|---|---|---|
| [affable-shark](https://bioimage.io/#/?id=affable-shark) | NucleiSegmentationBoundaryModel | 2D nuclei (DSB) | `bcyx (1, 1, 256, 256)` | `torchscript` | `(1, 2, 256, 256)` | `default` |
| [hiding-tiger](https://bioimage.io/#/?id=hiding-tiger) | LiveCellSegmentationBoundaryModel | 2D live-cell label-free (LIVECell) | `bcyx (1, 1, 256, 256)` | `torchscript` | `(1, 2, 256, 256)` | `default` |
| [powerful-chipmunk](https://bioimage.io/#/?id=powerful-chipmunk) | CovidIFCellSegmentationBoundaryModel | 2D cell (COVID-19 IF) | `bcyx (1, 1, 256, 256)` | `torchscript` | `(1, 2, 256, 256)` | `default` |
| [loyal-parrot](https://bioimage.io/#/?id=loyal-parrot) | HPA Cell Segmentation (DPNUnet) | 2D HPA cell segmentation | `bcyx (1, 3, 256, 256)` | `torchscript` | `(1, 3, 256, 256)` | `default` |
| [conscientious-seashell](https://bioimage.io/#/?id=conscientious-seashell) | HPA Nucleus Segmentation (DPNUnet) | 2D HPA nucleus segmentation | `bcyx (1, 3, 256, 256)` | `torchscript` | `(1, 3, 256, 256)` | `default` |
| [hiding-blowfish](https://bioimage.io/#/?id=hiding-blowfish) | EnhancerMitochondriaEM2D | 2D EM mitochondria boundary | `bcyx (1, 1, 256, 256)` | `torchscript` | `(1, 2, 256, 256)` | `default` |
| [shivering-raccoon](https://bioimage.io/#/?id=shivering-raccoon) | MitchondriaEMSegmentation2D | 2D EM mitochondria segmentation | `bcyx (1, 1, 256, 256)` | `torchscript` | `(1, 2, 256, 256)` | `default` |
| [pioneering-rhino](https://bioimage.io/#/?id=pioneering-rhino) | 2D UNet Arabidopsis Ovules | 2D plant Arabidopsis ovules | `bcyx (1, 1, 512, 512)` | `torchscript` | `(1, 1, 512, 512)` | `default` |
| [kind-seashell](https://bioimage.io/#/?id=kind-seashell) | MitochondriaEMSegmentationBoundaryModel | 3D EM mitochondria boundary | `bczyx (1, 1, 16, 128, 128)` | `torchscript` | `(1, 2, 16, 128, 128)` | `default` |
| [organized-badger](https://bioimage.io/#/?id=organized-badger) | PlatynereisEMnucleiSegmentationBoundaryModel | 3D EM nuclei (Platynereis) | `bczyx (1, 1, 32, 128, 128)` | `torchscript` | `(1, 2, 32, 128, 128)` | `default` |
| [willing-hedgehog](https://bioimage.io/#/?id=willing-hedgehog) | PlatynereisEMcellsSegmentationBoundaryModel | 3D EM cells (Platynereis) | `bczyx (1, 1, 32, 128, 128)` | `torchscript` | `(1, 1, 32, 128, 128)` | `default` |
| [powerful-fish](https://bioimage.io/#/?id=powerful-fish) | 3D UNet Mouse Embryo Live | 3D mouse embryo (light-sheet, live) | `bczyx (1, 1, 32, 128, 128)` | `torchscript` | `(1, 1, 32, 128, 128)` | `default` |
| [loyal-squid](https://bioimage.io/#/?id=loyal-squid) | 3D UNet Mouse Embryo Fixed | 3D mouse embryo (light-sheet, fixed) | `bczyx (1, 1, 32, 128, 128)` | `torchscript` | `(1, 2, 32, 128, 128)` | `default` |
| [emotional-cricket](https://bioimage.io/#/?id=emotional-cricket) | 3D UNet Arabidopsis Apical Stem Cells | 3D plant apical stem cells | `bczyx (1, 1, 100, 128, 128)` | `torchscript` | `(1, 1, 100, 128, 128)` | `default` |
| [thoughtful-turtle](https://bioimage.io/#/?id=thoughtful-turtle) | 3D UNet Lateral Root Primordia Cells | 3D plant lateral root (light-sheet) | `bczyx (1, 1, 100, 128, 128)` | `torchscript` | `(1, 1, 100, 128, 128)` | `default` |
| [noisy-fish](https://bioimage.io/#/?id=noisy-fish) | 3D UNet Arabidopsis Ovules Nuclei | 3D plant ovule nuclei | `bczyx (1, 1, 96, 96, 96)` | `torchscript` | `(1, 1, 96, 96, 96)` | `default` |
| [noisy-hedgehog](https://bioimage.io/#/?id=noisy-hedgehog) | 2D UNet label-free mCherry from BF | 2D label-free fluorescence prediction | `bcyx (1, 1, 512, 512)` | `torchscript` | `(1, 1, 512, 512)` | `default` |
| [ambitious-ant](https://bioimage.io/#/?id=ambitious-ant) | UniFMIRSuperResolutionOnMicrotubules | 2D super-resolution (microtubules) | `bcyx (1, 1, 128, 128)` | `torchscript` | `(1, 1, 256, 256)` | `default` |
| [courteous-otter](https://bioimage.io/#/?id=courteous-otter) | UniFMIRSuperResolutionOnFactin | 2D super-resolution (F-actin) | `bcyx (1, 1, 128, 128)` | `torchscript` | `(1, 1, 256, 256)` | `default` |
| [organized-cricket](https://bioimage.io/#/?id=organized-cricket) | Mitochondria SR (Wasserstein) | 2D mitochondria super-resolution | `bcxy (1, 1, 128, 128)` | `torchscript` | `(1, 1, 512, 512)` | `default` |
| [polite-pig](https://bioimage.io/#/?id=polite-pig) | HPA Bestfitting Densenet | HPA protein-localization classifier | `bcyx (1, 4, 512, 512)` | `onnx` | `(1, 28)` | `default` |
| [jolly-ox](https://bioimage.io/#/?id=jolly-ox) | MouseNuclei_N2V | 2D denoising (Noise2Void, mouse nuclei) | `bcyx (1, 1, 128, 128)` | `pytorch_state_dict` | `(1, 1, 128, 128)` | `with-careamics` |
| [sincere-microbe](https://bioimage.io/#/?id=sincere-microbe) | CHO mitotic rounding segmentation - brightfield - StarDist | 2D StarDist instance segmentation (CHO, brightfield) | `byxc (1, 256, 256, 1)` | `tensorflow_saved_model_bundle` | varies (StarDist heads) | `with-stardist` |

Two BIMZ landscape notes worth flagging: (1) most pre-2024 entries
publish only `pytorch_state_dict` plus optionally `torchscript` — `onnx`
is rare (1 of 21 here, `polite-pig`); the wrap's automatic
`onnx → torchscript → pytorch_state_dict` fallback is what makes the
`default` variant cover this list. (2) StarDist entries that publish
TensorFlow 1.15 SavedModels (`chatty-frog`, `fearless-crab`,
`modest-octopus`) cannot be loaded by the unstable-channel TF (2.21);
the `with-stardist` variant ships StarDist + TF 2.21 + Keras 3 + tf-keras
2 and is appropriate for newer StarDist entries (TF >= 2.x SavedModels,
e.g. `sincere-microbe`).

## Flake variants

The same `server.py` is exposed through four `nix run` flavors that differ
only in the Python environment they ship. Pick the smallest one that works
for your model.

| App                  | Adds                                                  | Use when…                          |
|----------------------|-------------------------------------------------------|------------------------------------|
| `.#default`          | nothing extra                                         | The model publishes ONNX or TorchScript weights (most of the zoo). |
| `.#with-stardist`    | `stardist` 0.9.2 + `csbdeep` 0.8.2 + `keras` 3 + `tf-keras` 2 (TF 2.21) | The RDF requires the StarDist Python package and a recent TF 2.x SavedModel. |
| `.#with-careamics`   | `careamics` 0.1.0 + `microssim` 0.0.3 (PyTorch / Lightning) | The RDF lists CAREamics (Noise2Void, CARE, etc.) as a runtime dependency. |
| `.#with-monai`       | `monai`                                               | The RDF requires a MONAI architecture. |

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
- `nix/careamics.nix`, `nix/microssim.nix` — from-source PyPI builds for
  `apps.with-careamics`. CAREamics is not in nixpkgs-unstable; microssim
  is one of its hard dependencies that also isn't.
- `nix/stardist.nix`, `nix/csbdeep.nix` — from-source PyPI builds for
  `apps.with-stardist`. csbdeep is patched to use `tf_keras` for its
  Keras-version probe (see `postPatch` in `nix/csbdeep.nix`) so it
  cooperates with the Keras 3 install nixpkgs ships alongside TF 2.21.
- `basic_test.py` — standalone smoke test (loads `affable-shark`, runs
  forward pass, asserts cuda + sensible output shape).
