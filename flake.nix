{
  description = "Flake for developing Super Resolution Models";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }: flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in {
      devShell = pkgs.mkShell {
        buildInputs = with pkgs; [
          gcc
          opencv4
          pkg-config
          gtk3
          gtk3-x11
        ];

        shellHook = ''
          echo "Welcome to the development environment!"
          echo "OpenCV version: $(pkg-config --modversion opencv4)"

          export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ 
            pkgs.libglvnd
            pkgs.opencv4
            pkgs.gtk3
          ]}:$LD_LIBRARY_PATH

          # Set up pkg-config path for OpenCV
          export PKG_CONFIG_PATH=${pkgs.opencv4}/lib/pkgconfig:$PKG_CONFIG_PATH
          export CPLUS_INCLUDE_PATH="$(pkg-config --cflags opencv4 | sed 's/-I//g'):$CPLUS_INCLUDE_PATH"

          USER_SHELL=$(getent passwd $USER | cut -d: -f7)
          exec $USER_SHELL
        '';
      };
    }
  );
}
