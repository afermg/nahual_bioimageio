{
  lib,
  buildPythonPackage,
  fetchPypi,
  setuptools,
  wheel,
  numpy,
  scikit-image,
  numba,
  imageio,
  csbdeep,
  llvmPackages,
}:
buildPythonPackage rec {
  pname = "stardist";
  version = "0.9.2";
  format = "setuptools";

  src = fetchPypi {
    inherit pname version;
    sha256 = "sha256-ZRIP4KCSYI89dvxmTGfN8UfQKsShAZN/uAamCnvLBVo=";
  };

  # OpenMP support requires libomp (the setup.py probes for -fopenmp).
  nativeBuildInputs = [
    llvmPackages.openmp
  ];

  build-system = [
    setuptools
    wheel
  ];

  dependencies = [
    csbdeep
    numpy
    scikit-image
    numba
    imageio
  ];

  doCheck = false;
  pythonRuntimeDepsCheck = false;
  dontCheckRuntimeDeps = true;

  pythonImportsCheck = [
    # Skip — importing stardist drags in tensorflow/csbdeep at module level
    # which would try to allocate CUDA in the build sandbox.
  ];

  meta = {
    description = "StarDist - Object Detection with Star-convex Shapes";
    homepage = "https://github.com/stardist/stardist";
    license = lib.licenses.bsd3;
  };
}
