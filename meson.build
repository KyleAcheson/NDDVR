project('fast_dvr', 'c',
  version : '0.1.0',
  license: 'BSD-3',
  meson_version: '>=0.64.0',
  default_options : ['warning_level=2'],
)

add_languages('fortran')

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

incdir_numpy = run_command(py,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

incdir_f2py = run_command(py,
    ['-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

h2opes_source = custom_target('h2opes.c',
  input : ['src/h2opes.f'],  # .f so no F90 wrappers
  output : ['h2opesmodule.c', 'h2opes-f2pywrappers.f'],
  command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'h2opes', '--lower']
)

nh3pes_source = custom_target('nh3pes.c',
  input : ['src/nh3pes.f'],  # .f so no F90 wrappers
  output : ['nh3pesmodule.c', 'nh3pes-f2pywrappers.f'],
  command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'nh3pes', '--lower']
)

inc_np = include_directories(incdir_numpy, incdir_f2py)

py.install_sources(
    ['src/__init__.py', 'src/dvr.py', 'src/exact_solvers.py', 'src/grids.py',
    'src/operators.py', 'src/plotting.py', 'src/potentials.py',
    'src/synthesised_solvers.py', 'src/transforms.py', 'src/wf_utils.py'],
    subdir: 'fast_dvr',
    pure: false,
)

py.extension_module('h2opes',
  ['src/h2opes.f', h2opes_source],
  incdir_f2py / 'fortranobject.c',
  include_directories: inc_np,
  subdir: 'fast_dvr',
  dependencies : py_dep,
  install : true
)

py.extension_module('nh3pes',
  ['src/nh3pes.f', nh3pes_source],
  incdir_f2py / 'fortranobject.c',
  include_directories: inc_np,
  subdir: 'fast_dvr',
  dependencies : py_dep,
  install : true
)
