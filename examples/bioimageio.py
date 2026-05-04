"""
This example uses a server within the environment defined on
`https://github.com/afermg/nahual_bioimageio.git`.

A single Nahual server fronts the entire BioImage Model Zoo. Pass the model
identifier (DOI / Zenodo URL / nickname / local rdf.yaml path) at setup()
time. The server picks the right backend (PyTorch / TensorFlow / ONNX) and
applies the RDF-declared pre/postprocessing.

Run the server first:
    nix run github:afermg/nahual_bioimageio -- ipc:///tmp/bioimageio.ipc

For RDFs that need extra Python deps (StarDist, CAREamics, MONAI), pick a
flavored variant:
    nix run github:afermg/nahual_bioimageio#with-stardist  -- ipc:///tmp/bioimageio.ipc
    nix run github:afermg/nahual_bioimageio#with-careamics -- ipc:///tmp/bioimageio.ipc
    nix run github:afermg/nahual_bioimageio#with-monai     -- ipc:///tmp/bioimageio.ipc
"""

import numpy

from nahual.process import dispatch_setup_process

# Not in nahual's built-in registry yet -> pass the signature explicitly.
setup, process = dispatch_setup_process("bioimageio", signature=("dict", "numpy"))
address = "ipc:///tmp/bioimageio.ipc"

# %% Load model server-side: any BIMZ identifier works.
parameters = dict(
    source="affable-shark",       # nickname; also accepts DOIs / Zenodo URLs / local paths
    # weight_format="onnx",       # optional; default tries onnx, then torchscript
    # device=0,                   # optional CUDA index
)
response = setup(parameters, address=address)
print(response)
# Expected (approx):
# {'device': 'cuda:0', 'model_id': '10.5281/zenodo.5764892/...',
#  'name': 'NucleiSegmentationBoundaryModel', 'input_axes': 'bcyx',
#  'weight_format': 'onnx', ...}

# %% Define data — shape must match the RDF's input axes.
# affable-shark is bcyx, 1 channel, 256x256.
numpy.random.seed(seed=42)
data = numpy.random.random_sample((1, 1, 256, 256)).astype("float32")
result = process(data, address=address)
print(result.shape)
# Expected: (1, 1, 256, 256) — segmentation mask, same spatial dims.
