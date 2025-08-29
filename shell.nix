{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "deep-learning-super-resolution";
  
  buildInputs = with pkgs; [
    # Python and core packages
    python313
    python313Packages.pip
    python313Packages.virtualenv
    
    # System libraries that OpenCV needs
    zlib
    glib
    xorg.libX11
    xorg.libXext
    xorg.libXrender
    gtk3
    
    # Python packages available in nixpkgs
    python313Packages.numpy
    python313Packages.opencv4
    python313Packages.pillow
    python313Packages.tkinter
    python313Packages.flask
  ];
  
  shellHook = ''
    echo "Setting up Deep Learning Super Resolution environment..."
    
    # Set up library paths for OpenCV
    export LD_LIBRARY_PATH="${pkgs.zlib}/lib:${pkgs.glib}/lib:${pkgs.xorg.libX11}/lib:${pkgs.xorg.libXext}/lib:${pkgs.xorg.libXrender}/lib:${pkgs.gtk3}/lib:$LD_LIBRARY_PATH"
    cd aaaa    
    # Create and activate virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
      echo "Creating virtual environment..."
      python -m venv .venv
    fi
    
    source .venv/bin/activate
    
    # Install packages not available in nixpkgs
    echo "Installing additional Python packages..."
    pip install sewar
    
    echo "Environment ready! You can now run: python main.py"
  '';
}
