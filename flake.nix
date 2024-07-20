{
  description = "";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" ];

      forEachSystem = with nixpkgs.lib; f: foldAttrs mergeAttrs { }
        (map (s: mapAttrs (_: v: { ${s} = v; }) (f s)) systems);
    in
    forEachSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        lib = pkgs.lib;

        pythonPackages = pkgs.python310.pkgs;

        libraries = [
          pkgs.zlib
          pkgs.stdenv.cc.cc.lib
        ];

        dependencies = [
          pythonPackages.venvShellHook
          pkgs.gnumake
        ];

        # some non-essential deps for linting, etc.
        dev-dependencies = [
          pkgs.ruff
          pythonPackages.pylsp-rope
          pythonPackages.python-lsp-ruff
          pythonPackages.python-lsp-server
          pythonPackages.python-lsp-server.optional-dependencies.all
          pythonPackages.ipython
        ];

        common_shell_attributes = {
          venvDir = ".venv";

          postVenv = ''
            unset SOURCE_DATE_EPOCH
          '';

          postShellHook = ''
            unset SOURCE_DATE_EPOCH
          '';

          LD_LIBRARY_PATH = lib.makeLibraryPath libraries;
          KERAS_BACKEND = "tensorflow";
        };

      in
      {
        devShells.default = pkgs.mkShell (common_shell_attributes // {
          packages = [
            dependencies
          ];
        });

        devShells.dev = pkgs.mkShell (common_shell_attributes // {
          packages = [
            dependencies
            dev-dependencies
          ];
        });
      });
}
