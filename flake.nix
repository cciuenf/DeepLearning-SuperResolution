{
  description = "Ambiente para manipulação de imagem com OpenCV e exibição com Tkinter";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python3Full;

        # Nosso ambiente Python com tudo que precisamos
        pythonWithPackages = python.withPackages (ps: with ps; [
          pip
          numpy      # Dependência do OpenCV e para manipulação de arrays
          opencv4    # Apenas o motor de processamento, sem GUI
          pillow     # Essencial para a "ponte" entre NumPy e Tkinter
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pythonWithPackages
          ];

          shellHook = ''
            echo "Ambiente pronto!"
            echo "Use OpenCV para processar e outra lib (ex: Tkinter) para exibir."
          '';
        };
      }
    );
}
