## Finite Quadratic Modules and Weil Representations ##

This repository contains a python package `fqm_weil`
to use in SageMath for working with finite quadratic modules and weil representations. 

## Requirements
- SageMath v9.7 - 10.0 (https://www.sagemath.org/)

## Installation

### Using sage pip
This package is not yet on PyPi. Please see manual installation instructions below.

### From git source
If the SageMath executable `sage` is in the current path you can install from source using the Makefile

```console
$ git clone https://github.com/fredstro/fqm-weil.git
$ cd fqm-weil
$ make install
```

### Docker
If you do not have SageMath installed, but you have docker you can use install this package
in a docker container built and executed using e.g. `make docker-sage` or `make docker-examples`

At the moment the `sagemath/sagemath:latest` image is using SageMath v9.7. 

## Main Contributors / Authors 
- H. Boylan
- S. Ehlen
- N.-P. Skoruppa
- F. Stromberg

- The initial version of this package was developed by Skoruppa and Boylan and a group of students in Siegen. 
Further work was then made by Ehlen and Stromberg and the current implementation and structure 
was mainly done by Stromberg. 
For a more detailed list see the list of authors in modules/finite_quadratic_module/finite_quadratic_module_base.py

## Usage

It is possible to construct a finite quadratic corresponding to a genus symbol:
    
    sage: from fqm_weil.all import FiniteQuadraticModule
    sage: F = FiniteQuadraticModule('7^-1.3.2_3^-1'); F
    Finite quadratic module in 3 generators:
    gens: e0, e1, e2
    form: 3/7*x0^2 + 2/3*x1^2 + 3/4*x2^2

We can study its Jordan decomposition:
    
    sage: F.jordan_decomposition()
    Jordan decomposition with genus symbol '2_3^-1.3.7^-1'

And decompose it further into indecomposable modules

    sage: F.jordan_decomposition().decompose()
    [2_3^-1, 3, 7^-1]

    sage: FiniteQuadraticModule('8_1^+3').jordan_decomposition().decompose()
    [8_1, 8^2]

For more usage examples see the docstrings of classes and functions. 

For more examples see the embedded doctests (search for `EXAMPLES`) as well as
the `/examples` directory which contains Jupyter notebook with an example of lifting maps
from scalar to vector-valued modular forms corresponding to the paper "On Liftings of modular forms and Weil representations by F. Stromberg."

## Examples

The directory `/examples` contains Jupyter notebooks with example code to illustrate the interface and functionality of this package. 
You can either open them manually from SageMath or run one of the following commands:
`make examples`
`make docker-examples`
which will start up a Jupyter notebook server from sagemath either locally or in a docker container. 

## Community Guidelines

### How to Contribute?
- Open an issue on GitHub and create a pull / merge request against the `develop` branch.
### How to report an issue or a problem? 
- First check if the issue is resolved in the `develop` branch. If not, open an issue on GitHub. 
### How to seek help and support?
- Contact the maintainer, Fredrik Stromberg, at: fredrik314@gmail.com (alternatively at fredrik.stromberg@nottingham.ac.uk)

## Development and testing

The make file `Makefile` contains a number of useful commands that you can run using 
```console
$ make <command>
```
The following commands are run in your local SagMath environment:
1. `build` -- builds the package in place (sometimes useful for development).
2. `sdist` -- create a source distribution in /sdist (can be installed using `sage -pip install sdist/<dist name>`)
3. `install` -- build and install the package in the currently active sage environment
4. `clean` -- remove all build and temporary files
5. `test` -- run sage's doctests (same as `sage -t src/*`)
6. `examples` -- run a Jupyter notebook with the SageMath kernel initialised at the `/examples` directory.
7. `tox` -- run `sage -tox` with all environments: `doctest`, `coverage`, `pycodestyle`, `relint`, `codespell`
   Note: If your local SageMath installation does not contain tox this will run `sage -pip install tox`.

The following commands are run in an isolated docker container 
and requires docker to be installed and running:
1. `docker` -- build a docker container with the tag `fqm_weil-{GIT_BRANCH}`
2. `docker-rebuild` -- rebuild the docker container without cache
3. `docker-test` -- run SageMath's doctests in the docker container
4. `docker-examples` -- run a Jupyter notebook with the SageMath kernel initialised at the `/examples` directory 
  and exposing the notebook at http://127.0.0.1:8888. The port used can be modified by the `NBPORT` parameter
5. `docker-tox` -- run tox with all environments: `doctest`, `coverage`, `pycodestyle`, `relint`, `codespell`. 
6. `docker-shell` -- run a shell in a docker container
7. `docker-sage` -- run a sage interactive shell in a docker container

The following command-line parameters are available 
- `NBPORT` -- set the port of the notebook for `examples` and `docker-examples`  (default is 8888)
- `TOX_ARGS` -- can be used to select one or more of the tox environments (default is all)
- `REMOTE_SRC` -- set to 0 if you want to use the local source instead of pulling from gitHub (default 1)
- `GIT_BRANCH` -- the branch to pull from gitHub (used if REMOTE_SRC=1)

### Example usage
Run tox coverage on the branch `main` from gitHub:

`make docker-tox REMOTE_SRC=1 GIT_BRANCH=main TOX_ARGS=coverage`

Run doctests on the local source with local version of sage:

`make tox TOX_ARGS=doctest`

Run relint on the local source with docker version of sage:

`make docker-tox REMOTE_SRC=0 TOX_ARGS=relint`

## Development

TODO

### GitHub Workflow

- There are two long-lived branches `main` and `develop`.
- The `develop` branch is used for development and can contain new / experimental features.  
- Pull-requests should be based on `develop`.
- Releases should be based on `main`.
- The `main` branch should always be as stable and functional as possible. In particular, merges should always happen from `develop` into `main`. 
- Git-Flow is enabled (and encouraged) with feature branches based on `develop` and hotfixes based on `main`. 
