{
  description = "Nahual server exposing the entire BioImage Model Zoo (BIMZ).";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
    pynng-flake.url = "github:afermg/pynng";
    pynng-flake.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      systems,
      ...
    }@inputs:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          system = system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        # ---- python packages we package ourselves (not yet in nixpkgs) ----
        # We also drop imagecodecs' very long pytest suite (it can take an
        # hour-plus on python3.13 and isn't relevant to our usage of
        # bioimageio).
        pyOverrides = pyfinal: pyprev: {
          genericache = pyfinal.callPackage ./nix/genericache.nix { };
          bioimageio-spec = pyfinal.callPackage ./nix/bioimageio_spec.nix { };
          bioimageio-core = pyfinal.callPackage ./nix/bioimageio_core.nix {
            bioimageio-spec = pyfinal.bioimageio-spec;
          };
          imagecodecs = pyprev.imagecodecs.overridePythonAttrs (_: {
            doCheck = false;
            doInstallCheck = false;
          });
        };

        python = pkgs.python3.override {
          packageOverrides = pyOverrides;
          self = python;
        };

        # ---- shared base set: what every variant gets ----
        baseDeps = pp: [
          packages.nahual
          pp.bioimageio-core
          pp.onnxruntime
          pp.torch
          pp.torchvision
          pp.numpy
          pp.xarray
          pp.scipy
          pp.tifffile
          pp.imageio
          pp.tqdm
        ];

        # ---- helpers to assemble an `apps.<name>` entry ----
        makePythonEnv = extras: python.withPackages (pp: baseDeps pp ++ extras pp);

        makeApp = name: extras:
          let
            python_with_pkgs = makePythonEnv extras;
            runServer = pkgs.writeScriptBin "runserver.sh" ''
              #!${pkgs.bash}/bin/bash
              ${python_with_pkgs}/bin/python ${self}/server.py ''${@:-"ipc:///tmp/bioimageio.ipc"}
            '';
          in
          {
            type = "app";
            program = "${runServer}/bin/runserver.sh";
          };

        packages = {
          nahual = python.pkgs.callPackage ./nix/nahual.nix {
            pynng = inputs.pynng-flake.packages.${system}.pynng;
          };
        };
      in
      with pkgs;
      {
        inherit packages;

        # ---- four flavored apps, all wiring to the SAME server.py ----
        apps = {
          # Plain bioimageio.core: enough for any RDF whose weights are
          # ONNX / TorchScript / pytorch_state_dict (most of the zoo).
          default = makeApp "default" (pp: [ ]);

          # Tier-2 add-ons for RDFs that ship custom architectures.
          with-stardist = makeApp "with-stardist" (pp: [ pp.stardist ]);
          with-careamics = makeApp "with-careamics" (pp: [ pp.careamics ]);
          with-monai = makeApp "with-monai" (pp: [ pp.monai ]);
        };

        devShells = {
          default =
            let
              python_with_pkgs = makePythonEnv (pp: [
                pp.scikit-image
                pp.scikit-learn
                pp.pyyaml
              ]);
            in
            mkShell {
              packages = [
                python_with_pkgs
                pkgs.cudaPackages.cudatoolkit
                pkgs.cudaPackages.cudnn
              ];
              shellHook = ''
                export PYTHONPATH=${python_with_pkgs}/${python_with_pkgs.sitePackages}:$PYTHONPATH
              '';
            };
        };
      }
    );
}
