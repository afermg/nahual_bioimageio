{
  description = "Nahual server exposing the entire BioImage Model Zoo (BIMZ).";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
    nahual-flake.url = "github:afermg/nahual";
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
          microssim = pyfinal.callPackage ./nix/microssim.nix { };
          careamics = pyfinal.callPackage ./nix/careamics.nix {
            bioimageio-core = pyfinal.bioimageio-core;
            microssim = pyfinal.microssim;
          };
          csbdeep = pyfinal.callPackage ./nix/csbdeep.nix { };
          stardist = pyfinal.callPackage ./nix/stardist.nix {
            csbdeep = pyfinal.csbdeep;
          };
          imagecodecs = pyprev.imagecodecs.overridePythonAttrs (_: {
            doCheck = false;
            doInstallCheck = false;
          });
          # nixpkgs propagates orbax-checkpoint (training/serialization helpers
          # that pull in jax-with-cuda) through keras 3. We only need keras 3
          # for the runtime path bioimageio.core takes
          # (tf.keras.layers.TFSMLayer ↔ Keras 3) — orbax + jax-cuda would add
          # several hours of from-source builds for nothing. Drop the heavy
          # extras here so the with-stardist closure stays bounded by TF +
          # stardist + csbdeep (the actually-used packages).
          keras = pyprev.keras.overridePythonAttrs (old:
            let
              drop = p: !(builtins.elem (p.pname or "") [
                "orbax-checkpoint"
                "tf2onnx"
                "scikit-learn"
              ]);
            in
            {
              dependencies = builtins.filter drop (old.dependencies or [ ]);
              propagatedBuildInputs = builtins.filter drop (old.propagatedBuildInputs or [ ]);
              doCheck = false;
              pythonImportsCheck = [ ];
              dontCheckRuntimeDeps = true;
            }
          );
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
          # nahual recipe sourced from upstream flake input; built against
          # our local python so it shares the override scope (keras drops,
          # imagecodecs check skip, etc).
          nahual = python.pkgs.callPackage (inputs.nahual-flake + "/nix/nahual.nix") { };
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
          # with-stardist also bundles Keras 3 + tf-keras 2 because:
          #  - bioimageio.core's KerasModelAdapter calls
          #    `tf.keras.layers.TFSMLayer(...)`, which on TF >= 2.16 lazy-loads
          #    Keras 3 (tf.keras becomes the standalone `keras` package);
          #  - csbdeep's `from keras import __version__` probe is patched to
          #    look at tf_keras (Keras 2) instead so its post-TF-2.6 code path
          #    doesn't blow up against the installed Keras 3.
          with-stardist = makeApp "with-stardist" (pp: [
            pp.stardist
            pp.keras
            pp.tf-keras
          ]);
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
                # python.withPackages already wraps python with the right
                # site-packages on the import path, so an explicit PYTHONPATH
                # is redundant.
              '';
            };
        };
      }
    );
}
