#from setuptools import setup
from numpy.distutils.core import setup
from numpy.distutils.core import Extension

ext1 = Extension(name='NemesisPy.nemesisf',
                 sources=['NemesisPy/Fortran/mulscatter.f90','NemesisPy/Fortran/spectroscopy.f90','NemesisPy/Fortran/hapke.f90'],
                 f2py_options=['--quiet'],
                 )

setup(name='NemesisPy',
      version='0.0.1',
      description='NEMESIS radiative transfer code',
      packages=['NemesisPy'],
      install_requires=['numpy','matplotlib','sympy','miepython','numba','ray','scipy'],
      ext_modules=[ext1],
      )
