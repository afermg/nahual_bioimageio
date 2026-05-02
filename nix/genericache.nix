{
  lib,
  buildPythonPackage,
  fetchPypi,
  setuptools,
  wheel,
  filelock,
}:
buildPythonPackage rec {
  pname = "genericache";
  version = "0.5.2";
  format = "pyproject";

  src = fetchPypi {
    inherit pname version;
    sha256 = "sha256-G9whsEcf/TCSx4277ix1etREuqdGBzxoBo8PQVCdgm8=";
  };

  build-system = [
    setuptools
    wheel
  ];

  dependencies = [
    filelock
  ];

  pythonImportsCheck = [
    "genericache"
  ];

  doCheck = false;
  pythonRuntimeDepsCheck = false;
  dontCheckRuntimeDeps = true;

  meta = {
    description = "Generic cache for Python objects on disk";
    homepage = "https://pypi.org/project/genericache/";
    license = lib.licenses.mit;
  };
}
