{
  description = "C++ project with nix flake";

  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let pkgs = import nixpkgs { inherit system; }; in
        {
          devShell = pkgs.mkShell {
            name = "cpp-build-toolchain";

            nativeBuildInputs = [ pkgs.llvmPackages_21.clang ];

            buildInputs = with pkgs; [
              cmake
              ninja

              llvmPackages_21.libcxx
              llvmPackages_21.openmp
              llvmPackages_21.clang-tools

              gdb
              boost
              openblas
            ];

            # this exquisite crutch lets me reference absolute path to clang++
            # binary in settings.json like this: "--query-driver=${env:PROJECT_CC}/bin/clang++"
            # this is required since clangd needs absolute paths for these drivers and vscode can't
            # use command substitution in settings.json
            PROJECT_CC = "${pkgs.llvmPackages_21.clang}";
          };
        }
      );
}
