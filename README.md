# nahual_bioimageio

A single Nahual server fronting the entire **BioImage Model Zoo** (BIMZ): pass any RDF identifier (DOI, Zenodo URL, nickname like `affable-shark`, or a local `rdf.yaml` path) at `setup()` time, and the wrap loads it through `bioimageio.core`, picks the right backend (PyTorch / TensorFlow / ONNX), applies the RDF-declared pre/postprocessing, and gives you back numpy arrays.

## Validated BIMZ models

23 entries verified end-to-end on `cuda:0`. Click any nickname for its model card on `bioimage.io`. Run with:

```bash
nix run --impure github:afermg/nahual_bioimageio[#with-<variant>] -- ipc:///tmp/bioimageio.ipc
```

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
| [sincere-microbe](https://bioimage.io/#/?id=sincere-microbe) | CHO mitotic rounding (StarDist) | 2D StarDist instance seg (CHO brightfield) | `byxc (1, 256, 256, 1)` | `tensorflow_saved_model_bundle` | varies (StarDist heads) | `with-stardist` |

Picked from the BIMZ collection JSON by (a) high download count, (b) `onnx` or `torchscript` weights so the `default` flake variant covers them with no model-architecture deps, and (c) coverage spread across nuclei / cell / EM segmentation, super-resolution, label-free prediction, and classification. Three additional candidates were attempted but dropped: `joyful-deer` and `straightforward-crocodile` (bioimageio.core 0.10.2 hasn't implemented the v0_4 `ScaleLinearKwargs(axes=...)` postprocessing) and `nice-peacock` (3-input model — single-tensor wrap can't drive it).

**BIMZ landscape notes:**

- **TorchScript dominates.** Of the 21 default-variant entries, 20 use `torchscript` and only 1 (`polite-pig`) uses `onnx`. The wrap's automatic `onnx → torchscript → pytorch_state_dict` fallback is what makes `default` cover this list without per-model architecture deps.
- **TF 1.15 SavedModels are unloadable.** StarDist entries publishing TF 1.15 SavedModels (`chatty-frog`, `fearless-crab`, `modest-octopus`) can't load on this stack — `bioimageio.core 0.10.2` routes through Keras 3's `TFSMLayer`, which doesn't consume TF1 graph-def SavedModels. Use the dedicated [`afermg/stardist`](https://github.com/afermg/stardist) wrap (TF 2.13 / Python 3.11) for those.
- **`with-stardist` covers TF 2.x SavedModels** (e.g. `sincere-microbe`); `with-careamics` covers any RDF that depends on the CAREamics Python package (e.g. `jolly-ox`, the N2V family).

## What's new vs vanilla `bioimageio.core`

This wrap adds three layers on top:

1. **Nahual IPC.** Long-lived server, numpy in/out across process and environment boundaries — so a downstream pipeline doesn't have to inherit bioimageio.core's dep closure. With `nahual >= 0.0.9` you can `setup()` again on the same server to swap models without restarting the IPC daemon.
2. **GPU-first packaging.** `cudaSupport = true`; the dev shell ships `cudaPackages.cudatoolkit` + `cudnn`. ONNX Runtime resolves to its CUDA-built variant; PyTorch/TensorFlow paths land on `cuda:0` (or `GPU:0` for TF). Validation in this README is on a real GPU, not CPU smoke.
3. **Tier-2 flake variants for custom-architecture RDFs.** RDFs that publish only `pytorch_state_dict` need their architecture importable in the env. Three add-on variants share the same `server.py`:

| App | Adds | Use when… |
|---|---|---|
| `.#default` | nothing extra | The model publishes ONNX or TorchScript (most of the zoo). |
| `.#with-stardist` | `stardist` 0.9.2 + `csbdeep` 0.8.2 + `keras` 3 + `tf-keras` 2 (TF 2.21) | Recent StarDist RDFs (TF 2.x SavedModels). |
| `.#with-careamics` | `careamics` 0.1.0 + `microssim` 0.0.3 | RDF lists CAREamics as a runtime dep (CARE, N2V, etc.). |
| `.#with-monai` | `monai` | RDF requires a MONAI architecture. |

`careamics`, `microssim`, `stardist`, and `csbdeep` are packaged from PyPI as proper Nix derivations — no conda/micromamba bootstrap. `csbdeep` is `postPatch`-ed to look up the Keras version through `tf_keras` (rather than its raw `from keras import __version__`), which lets it cooperate with the Keras 3 install nixpkgs ships alongside TF 2.21.

## Run

```bash
# Default flavor.
nix run --impure github:afermg/nahual_bioimageio -- ipc:///tmp/bioimageio.ipc

# Or pick a flavor for your model:
nix run --impure github:afermg/nahual_bioimageio#with-stardist  -- ipc:///tmp/bioimageio.ipc
nix run --impure github:afermg/nahual_bioimageio#with-careamics -- ipc:///tmp/bioimageio.ipc
nix run --impure github:afermg/nahual_bioimageio#with-monai     -- ipc:///tmp/bioimageio.ipc
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
print(info)  # {'device': 'cuda:0', 'weight_format': 'torchscript', 'input_axes': 'bcyx', ...}

img = np.random.random_sample((1, 1, 256, 256)).astype("float32")  # bcyx
out = process(img, address="ipc:///tmp/bioimageio.ipc")
print(out.shape)  # (1, 2, 256, 256) for affable-shark
```

## `setup()` parameters

| Param | Type | Default | Notes |
|---|---|---|---|
| `source` | `str` | — | Required. Any identifier `bioimageio.core.load_description` accepts. |
| `weight_format` | `str \| None` | `None` | If `None`, tries `onnx` then `torchscript` then `pytorch_state_dict`. Errors informatively if no preferred format is published, listing what IS available. |
| `device` | `int` | `0` | CUDA device index → `devices=["cuda:<device>"]`. |

## `process()` contract

Pass a numpy array whose dims match the RDF's declared input axes (returned in the `input_axes` field of `setup()`'s response — e.g. `bcyx`). The wrap reshapes it into an `xarray.DataArray`, runs the prediction pipeline, and returns the first output as a plain numpy array. **No `NCZYX` normalization here** — different RDFs declare different axes; reshape on the client side to match.

## Files

- `server.py` — Nahual server, multi-backend.
- `flake.nix` — four apps (`default` / `with-stardist` / `with-careamics` / `with-monai`) plus a dev shell.
- `nix/bioimageio_core.nix`, `nix/bioimageio_spec.nix`, `nix/genericache.nix` — from-PyPI builds; nixpkgs doesn't ship these.
- `nix/careamics.nix`, `nix/microssim.nix` — from-PyPI builds for `apps.with-careamics`.
- `nix/stardist.nix`, `nix/csbdeep.nix` — from-PyPI builds for `apps.with-stardist`. `csbdeep` is patched to look up the Keras version through `tf_keras`.
- `basic_test.py` — standalone smoke test (loads `affable-shark`, asserts `cuda` + sensible output shape).
- `examples/bioimageio.py` — Nahual client demo.
