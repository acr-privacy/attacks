{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" ];

      forEachSystem = with nixpkgs.lib; f: foldAttrs mergeAttrs { }
        (map (s: mapAttrs (_: v: { ${s} = v; }) (f s)) systems);
    in
    forEachSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python38;
        pythonPackages = python.pkgs;
        libraries = [
          pkgs.stdenv.cc.cc.lib
        ];
      in
      {

        devShells.default = pkgs.mkShell {
          packages = [
            pythonPackages.venvShellHook
            pkgs.gnumake
            pkgs.gnutar
          ];

          venvDir = ".venv";

          postVenv = ''
            unset SOURCE_DATE_EPOCH
          '';

          postShellHook = ''
            # Allow the use of wheels.
            unset SOURCE_DATE_EPOCH
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath libraries}"
          '';

        };
      });
}
