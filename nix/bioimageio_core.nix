{
  lib,
  buildPythonPackage,
  fetchPypi,
  setuptools,
  pip,
  # runtime deps
  bioimageio-spec,
  imagecodecs,
  imageio,
  loguru,
  numpy,
  pydantic,
  pydantic-settings,
  ruyaml,
  scipy,
  tqdm,
  typing-extensions,
  xarray,
}:
buildPythonPackage rec {
  pname = "bioimageio.core";
  version = "0.10.2";
  format = "pyproject";

  src = fetchPypi {
    pname = "bioimageio_core";
    inherit version;
    sha256 = "sha256-OpX0vY531wJvljgz0dhM3ATifdAW/x6c7iagt8s6Q1s=";
  };

  build-system = [
    setuptools
    pip
  ];

  dependencies = [
    bioimageio-spec
    imagecodecs
    imageio
    loguru
    numpy
    pydantic
    pydantic-settings
    ruyaml
    scipy
    tqdm
    typing-extensions
    xarray
  ];

  pythonImportsCheck = [
    "bioimageio.core"
  ];

  doCheck = false;
  pythonRuntimeDepsCheck = false;
  dontCheckRuntimeDeps = true;

  meta = {
    description = "Run-time engine for BioImage Model Zoo (loads any RDF and runs prediction with the right backend)";
    homepage = "https://github.com/bioimage-io/core-bioimage-io-python";
    license = lib.licenses.mit;
  };
}
