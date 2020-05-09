# Building ARPACK 

## macOS

1. Make sure you have `gfortran` installed (if you installed gcc via Homebrew, then you already have it). Use `gfortran --version` to check.
2. Install some prerequisites using Homebrew: `brew install libtools automake pkg-config`
3. `make bootstrap-arpack`
4. `make configure-arpack`
5. `make arpack`

## Linux

TODO
