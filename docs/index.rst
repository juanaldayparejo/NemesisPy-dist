NemesisPy
=========

.. image:: https://img.shields.io/badge/readthedocs-latest-blue
   :target: https://nemesispy.readthedocs.io

.. image:: https://img.shields.io/badge/github-code-green
   :target: https://github.com/juanaldayparejo/NemesisPy-dist

.. image:: https://img.shields.io/badge/NEMESIS-reference-yellow
   :target: https://doi.org/10.1016/j.jqsrt.2007.11.006


__________

This website includes the documentation regarding the Python version of the NEMESIS (Non-linear Optimal Estimator for MultivariatE
Spectral analySIS) planetary atmosphere radiative transfer and retrieval code. 

The main description of the NEMESIS code was published by `Irwin et al. (2008) <https://doi.org/10.1016/j.jqsrt.2007.11.006>`_.
The original Fortran version of the code is `available here <https://doi.org/10.5281/zenodo.4303976>`_.

In this website, we aim to provide a more practical description of the code, including retrieval examples applied to different observing geometries or physical parameterisations.

**NOTE:** At this stage, documentation is under development.

Install NemesisPy
------------------

The latest version of code has to be downloaded from `Github <https://github.com/juanaldayparejo/NemesisPy-dist.git>`_.

Once the code has been downloaded from Github, move the NemesisPy-dist/ package to a desired path. Then, inside the package, type ::

$ pip install --editable .

This will install the NemesisPy package, but with the ability to update any changes made to the code (e.g., when introducing new model parameterisations).

In the future, we aim to release official versions to The Python Package Index (PyPI), so that it can be directly installed using pip.


Revision history
-----------------------------

- 0.0.0 (14 January, 2022)
    - First version of the code.

Dependencies on other Python packages
-----------------------------

- `numpy <https://numpy.org/>`_: Used widely throughout the code to define N-dimensional arrays and perform mathematical operations (e.g., matrix multiplication).
- `matplotlib <https://matplotlib.org/>`_: Used to create visualizations. 
- `miepython <https://miepython.readthedocs.io/en/latest/>`_: Used to calculate the optical properties of spherical particles using Mie Theory.
- `numba <https://numba.pydata.org/>`_: Used in specific functions to include the JIT compiler decorator and speed up the radiative transfer calculations.

.. toctree::
   :maxdepth: 2

.. toctree::
   :caption: Radiative transfer
   :hidden:
   
   radiative_transfer.ipynb
 
.. toctree::
   :caption: Retrievals
   :hidden:
   
   retrievals.ipynb
   
.. toctree::
   :caption: Examples
   :hidden:
   
   examples

.. toctree::
   :caption: Package Details
   :hidden:
   
   documentation


