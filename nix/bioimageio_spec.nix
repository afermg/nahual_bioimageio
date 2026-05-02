{
  lib,
  buildPythonPackage,
  fetchPypi,
  setuptools,
  pip,
  # runtime deps
  annotated-types,
  email-validator,
  exceptiongroup,
  genericache,
  httpx,
  imageio,
  loguru,
  markdown,
  numpy,
  packaging,
  platformdirs,
  pydantic,
  pydantic-core,
  pydantic-settings,
  python-dateutil,
  rich,
  ruyaml,
  tifffile,
  tqdm,
  typing-extensions,
  zipp,
}:
buildPythonPackage rec {
  pname = "bioimageio.spec";
  version = "0.5.9.1";
  format = "pyproject";

  src = fetchPypi {
    pname = "bioimageio_spec";
    inherit version;
    sha256 = "sha256-wWFcZD0Qi2CTats1YvgNxUbQyxTLQ5HF87g4pOg8ecE=";
  };

  build-system = [
    setuptools
    pip
  ];

  dependencies = [
    annotated-types
    email-validator
    exceptiongroup
    genericache
    httpx
    imageio
    loguru
    markdown
    numpy
    packaging
    platformdirs
    pydantic
    pydantic-core
    pydantic-settings
    python-dateutil
    rich
    ruyaml
    tifffile
    tqdm
    typing-extensions
    zipp
  ];

  pythonImportsCheck = [
    "bioimageio.spec"
  ];

  doCheck = false;
  pythonRuntimeDepsCheck = false;
  dontCheckRuntimeDeps = true;

  meta = {
    description = "Specifications for the BioImage Model Zoo (RDF parser/validator)";
    homepage = "https://github.com/bioimage-io/spec-bioimage-io";
    license = lib.licenses.mit;
  };
}
