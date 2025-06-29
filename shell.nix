
{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  name = "dev-shell";
  buildInputs = with pkgs.python313Packages; [
    numpy
    flask
    # …
    (opencv4.override { enableGtk2 = true; })
  ];
}

