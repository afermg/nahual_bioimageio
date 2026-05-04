{
  lib,
  buildPythonPackage,
  fetchPypi,
  hatchling,
  hatch-vcs,
  # runtime deps
  bioimageio-core,
  matplotlib,
  microssim,
  numpy,
  pillow,
  psutil,
  pydantic,
  pytorch-lightning,
  pyyaml,
  scikit-image,
  tifffile,
  torch,
  torchmetrics,
  torchvision,
  typer,
  validators,
  zarr,
}:
buildPythonPackage rec {
  pname = "careamics";
  version = "0.1.0";
  format = "pyproject";

  src = fetchPypi {
    inherit pname version;
    sha256 = "sha256-t6w0eHahFhidc7H5SEW1Ri6Y8nsfS0e5hqXz2nKA/vU=";
  };

  # hatch-vcs reads the version from git tags. The PyPI sdist has no .git
  # directory, so we hand it the version explicitly.
  env.HATCH_BUILD_HOOK_VCS_VERSION_FILE = "src/careamics/_version.py";
  env.SETUPTOOLS_SCM_PRETEND_VERSION = version;

  build-system = [
    hatchling
    hatch-vcs
  ];

  dependencies = [
    bioimageio-core
    matplotlib
    microssim
    numpy
    pillow
    psutil
    pydantic
    pytorch-lightning
    pyyaml
    scikit-image
    tifffile
    torch
    torchmetrics
    torchvision
    typer
    validators
    zarr
  ];

  pythonImportsCheck = [
    "careamics"
  ];

  doCheck = false;
  pythonRuntimeDepsCheck = false;
  dontCheckRuntimeDeps = true;

  meta = {
    description = "Toolbox for running N2V and friends through bioimageio.core";
    homepage = "https://github.com/CAREamics/careamics";
    license = lib.licenses.bsd3;
  };
}
