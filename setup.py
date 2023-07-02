import os
import shutil
import subprocess
import setuptools
from sage_setup.extensions import create_extension
from setuptools.extension import Extension
import Cython.Compiler.Main
from Cython.Build import cythonize
from sage.env import SAGE_LIB

debug = False
gdb_debug = True
if os.environ.get('SAGE_DEBUG', None) == 'yes':
    print('Enabling Cython debugging support')
    debug = True
    Cython.Compiler.Main.default_options['gdb_debug'] = True
    Cython.Compiler.Main.default_options['output_dir'] = 'build'
    gdb_debug = True

LIBRARY_DIRS = []
INCLUDE_DIRS = []
if shutil.which('brew') is not None:
    proc = subprocess.Popen("/opt/homebrew/bin/brew --prefix", shell=True,
                            stdout=subprocess.PIPE, stdin=subprocess.PIPE,
                            stderr=subprocess.STDOUT, close_fds=True)
    HOMEBREW_PREFIX = proc.stdout.readline().decode('utf-8').strip()
    HOMEBREW_LIB = HOMEBREW_PREFIX + '/lib'
    LIBRARY_DIRS.append(HOMEBREW_LIB)
    HOMEBREW_INC = HOMEBREW_PREFIX + '/include'
    INCLUDE_DIRS.append(HOMEBREW_INC)

INCLUDE_DIRS += ['src/fqm_weil/modules']
extra_compile_args = ['-Wno-unused-function',
                      '-Wno-implicit-function-declaration',
                      '-Wno-unused-variable',
                      '-Wno-deprecated-declarations',
                      '-Wno-deprecated-register',
                      '-Wno-unreachable-code',
                      '-Wno-unreachable-code-fallthrough']

ext_modules = [
    Extension(module, sources, include_dirs=INCLUDE_DIRS,
              extra_compile_args=extra_compile_args,
              library_dirs=LIBRARY_DIRS)
    for (module, sources) in [
        ('fqm_weil.modules.weil_module.weil_invariants',
         ['src/fqm_weil/modules/weil_module/weil_invariants.pyx']),
        ('fqm_weil.modules.weil_module.weil_module_alg',
         ['src/fqm_weil/modules/weil_module/weil_module_alg.pyx']),
        ('fqm_weil.modules.utils', ['src/fqm_weil/modules/utils.pyx'])
        ]
]

extensions = cythonize(
    ext_modules,
    include_path=['src', 'src/fqm_weil/modules'] + LIBRARY_DIRS + [SAGE_LIB],
    compiler_directives={
        'embedsignature': True,
        'language_level': '3',
    },
    gdb_debug=gdb_debug,
)

setuptools.setup(
    ext_modules=extensions,
    create_extension=create_extension,
    packages=['fqm_weil',
              'fqm_weil.modular',
              'fqm_weil.modules',
              'fqm_weil.modules.finite_quadratic_module',
              'fqm_weil.modules.weil_module',
              ],
    package_data={
        "": ["*.pxd"]
    }
)
