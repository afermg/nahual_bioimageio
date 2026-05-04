{
  lib,
  buildPythonPackage,
  fetchPypi,
  hatchling,
  hatch-vcs,
  numpy,
  scipy,
  scikit-image,
  torch,
  torchmetrics,
  tqdm,
}:
buildPythonPackage rec {
  pname = "microssim";
  version = "0.0.3";
  format = "pyproject";

  src = fetchPypi {
    inherit pname version;
    sha256 = "sha256-rXgX12yoUbeyjSZet5MXVtZOMIvLkECzFgBoV9LQskA=";
  };

  # hatch-vcs reads the version from git tags. The PyPI sdist has no .git
  # directory, so we hand it the version explicitly.
  env.HATCH_BUILD_HOOK_VCS_VERSION_FILE = "src/microssim/_version.py";
  env.SETUPTOOLS_SCM_PRETEND_VERSION = version;

  build-system = [
    hatchling
    hatch-vcs
  ];

  dependencies = [
    numpy
    scipy
    scikit-image
    torch
    torchmetrics
    tqdm
  ];

  pythonImportsCheck = [
    "microssim"
  ];

  doCheck = false;
  pythonRuntimeDepsCheck = false;
  dontCheckRuntimeDeps = true;

  meta = {
    description = "Improved structural similarity metrics for comparing microscopy data";
    homepage = "https://github.com/juglab/MicroSSIM";
    license = lib.licenses.bsd3;
  };
}
