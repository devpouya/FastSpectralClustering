# Building ARPACK 

## macOS

1. Make sure you have `gfortran` installed (if you installed gcc via Homebrew, then you already have it). Use `gfortran --version` to check. Otherwise `brew install gcc`.
2. Install some prerequisites using Homebrew: `brew install openblas libtools automake pkg-config cmake`
3. `make bootstrap-arpack`
4. `make configure-arpack`
5. `make arpack`

## Linux

1. Install gfortran: `sudo apt-get install gfortran`
2. Install some prerequisites using apt: `sudo apt-get install libopenblas-dev dh-autoreconf pkg-config cmake`
3. `make bootstrap-arpack`
4. `make configure-arpack`
5. `make arpack`

