
{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  name = "dev-shell";
  buildInputs = with pkgs.python313Packages; [
    numpy
    flask
    tkinter
    # …
    (opencv4.override { enableGtk2 = true; })
  ];
}

