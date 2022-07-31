## Finite Quadratic Modules and Weil Representations ##

This repository contains a python package `fqm_weil`
to use in SageMath for working with finite quadratic modules and weil representations. 

## Requirements
- SageMath v9.4+ (https://www.sagemath.org/)

## Installation

### Using sage pip
This package needs to be installed in the virtual environment provided by SageMath, and it is therefore necessary 
to run the following command 
```console
$ sage -pip install --no-build-isolation hilbert-modular-group
```
**Note**: The `--no-build-isolation` is necessary as the compiler needs access 
to certain library files from the sage installation and SageMath itself is 
too large to be required as a build dependency. 
As an alternative to this flag you can also specify the environment variable 
SAGE_LIB explicitly.

### From git source
If the SageMath executable `sage` is in the current path you can install from source using the Makefile

```console
$ git clone https://github.com/fredstro/fqm-weil.git
$ cd hilbertmodgrup
$ make install
```

### Docker
If you do not have SageMath installed, but you have docker you can use install this package
in a docker container built and executed using e.g. `make docker-sage` or `make docker-examples`


## Main Contributors / Authors
- N.-P. Skoruppa
- H. Boylan
- S. Ehlen
- F. Stromberg
 
For a more detailed list see the list of authors in modules/finite_quadratic_module.py