{
  description = "C++ project with nix flake";

  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem
      (system:
      let
        nvidia-pkgs-whitelist = [
          "cuda_cuobjdump"
          "cuda_gdb"
          "cuda_nvcc"
          "cuda_nvdisasm"
          "cuda_nvprune"
          "cuda_cccl"
          "cuda_cudart"
          "cuda_cupti"
          "cuda_cuxxfilt"
          "cuda_nvml_dev"
          "cuda_nvrtc"
          "cuda_nvtx"
          "cuda_profiler_api"
          "cuda_sanitizer_api"
          "libcublas"
          "libcufft"
          "libcurand"
          "libcusolver"
          "libnvjitlink"
          "libcusparse"
          "libnpp"
          "cuda-merged"
          "nsight_compute"
          "nsight_systems"
        ];

        common-pkgs = import nixpkgs {
          inherit system;
          # allow unfree predicate to allow mkl
          config.allowUnfreePredicate = pkg: builtins.elem (nixpkgs.lib.getName pkg) (nvidia-pkgs-whitelist ++ ["mkl"]);
        };

        build-inputs-list = pkgs-set: with pkgs-set; [
          cmake
          ninja

          llvmPackages_21.openmp
          llvmPackages_21.clang-tools

          cudaPackages_12.cudatoolkit
          cudaPackages_12.cuda_cudart
          cudaPackages_12.nsight_compute
          cudaPackages_12.nsight_systems

          gdb
          boost
          openblas
          mkl

          openmpi

          linuxPackages.perf
        ];

        native-build-inputs = [ common-pkgs.gcc14 ];
        build-inputs = build-inputs-list common-pkgs;
      in
        {
          devShell = common-pkgs.mkShell {
            name = "cpp-build-toolchain";

            nativeBuildInputs = native-build-inputs;
            buildInputs = build-inputs;

            # in order for prebuild binaries like python wheels and other packages
            # to operate, we need to patch the env a bit, including paths to runtime libs
            LD_LIBRARY_PATH = "${common-pkgs.lib.makeLibraryPath (build-inputs ++ native-build-inputs)}";

            # this exquisite crutch lets me reference absolute path to clang++
            # binary in settings.json like this: "--query-driver=${env:PROJECT_CC}/bin/clang++"
            # this is required since clangd needs absolute paths for these drivers and vscode can't
            # use command substitution in settings.json
            PROJECT_CC = "${common-pkgs.gcc14}/bin/g++";
          };
        }
      );
}
