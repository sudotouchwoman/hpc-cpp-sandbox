{
  description = "C++ project with nix flake";

  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let pkgs = nixpkgs.legacyPackages.${system}; in
        {
          devShells.default = pkgs.mkShell {
            name = "cpp-build-toolchain";
              buildInputs = with pkgs; [
                gnumake
                gcc14Stdenv
                cmake
                ninja
                gdb
                clang-tools
                boost
                openmpi
              ];

            shellHook = '''';
          };
        }
      );
}
