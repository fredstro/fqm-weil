import os
import setuptools
from setuptools.extension import Extension
from Cython.Build import cythonize
# Check if we are currently in a SageMath environment.
SAGE_LOCAL = os.getenv('SAGE_LOCAL')
if not SAGE_LOCAL:
    raise ValueError("This package can only be installed inside SageMath (http://www.sagemath.org)")
# Find correct value for SAGE_LIB which is needed to compile the Cython extensions.
SAGE_LIB = os.getenv('SAGE_LIB')
if not SAGE_LIB:
    try:
        from sage.env import SAGE_LIB
    except ModuleNotFoundError:
        raise ModuleNotFoundError("To install this package you need to either specify the "
                                  "environment variable 'SAGE_LIB' or call pip with "
                                  "'--no-build-isolation'")
if not os.path.isdir(SAGE_LIB):
    raise ValueError(f"The library path {SAGE_LIB} is not a directory.")
# SAGE_INC = SAGE_LOCAL + "/include"
# Extension modules using Cython
extra_compile_args = ['-Wno-unused-function',
                      '-Wno-implicit-function-declaration',
                      '-Wno-unused-variable',
                      '-Wno-deprecated-declarations',
                      '-Wno-deprecated-register']
ext_modules = [
    Extension(
        'fqm_weil.modules.weil_invariants',
        sources=['src/fqm_weil/modules/weil_invariants.pyx'],
        include_dirs=['src/fqm_weil/modules', '/usr/local/lib'],
        extra_compile_args=extra_compile_args
    ),
    Extension(
        'fqm_weil.modules.weil_module_alg',
        sources=['src/fqm_weil/modules/weil_module_alg.pyx'],
        include_dirs=['src/fqm_weil/modules'],
        extra_compile_args=extra_compile_args
    ),
    Extension(
        'fqm_weil.modules.utils',
        sources=['src/fqm_weil/modules/utils.pyx'],
        include_dirs=['src/fqm_weil/modules'],
        extra_compile_args=extra_compile_args
    )]
debug = False
gdb_debug = True
import Cython
if os.environ.get('SAGE_DEBUG', None) == 'yes':
    print('Enabling Cython debugging support')
    debug = True
    Cython.Compiler.Main.default_options['gdb_debug'] = True
    Cython.Compiler.Main.default_options['output_dir'] = 'build'
    gdb_debug = True


setuptools.setup(
    ext_modules=cythonize(
        ext_modules,
        include_path=['src', 'src/fqm_weil/modules', SAGE_LIB],
        compiler_directives={
            'embedsignature': True,
            'language_level': '3',
        },
        gdb_debug=gdb_debug,
    ),
    packages=['fqm_weil',
              'fqm_weil.modules',
              # 'fqm_weil.modules.finite_quadratic_module',
              # 'fqm_weil.modules.weil_module',
              ],
    package_data={
        "": ["*.pxd"]
    }
)
