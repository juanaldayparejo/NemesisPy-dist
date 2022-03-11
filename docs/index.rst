The NEMESIS radiative transfer and retrieval code
======================================================================

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

Publications using NEMESIS
-----------------------------

  - Alday, J., Wilson, C. F., Irwin, P. G. J., Trokhimovskiy, A., Montmessin, F., Fedorova, A. A., et al. (2021). Isotopic composition of CO2 in the atmosphere of Mars: Fractionation by diffusive separation observed by the ExoMars Trace Gas Orbiter. Journal of Geophysical Research: Planets, 126, e2021JE006992. https://doi.org/10.1029/2021JE006992
  
  - Alday, J., Trokhimovskiy, A., Irwin, P. G. J., Wilson, C. F., Montmessin, F., Lef vre, F., et al. (2021). Isotopic fractionation of water and its photolytic products in the atmosphere of Mars. Nature Astronomy, 5, 943–950. https://doi.org/10.1038/s41550-021-01389-x
  
  - Irwin, P. G., Parmentier, V., Taylor, J., Barstow, J., Aigrain, S., Lee, G. K., & Garland, R. (2020). 2.5 D retrieval of atmospheric properties from exoplanet phase curves: application to WASP-43b observations. Monthly Notices of the Royal Astronomical Society, 493(1), 106-125. https://doi.org/10.1093/mnras/staa238
  
  - Irwin, P. G. J., Toledo, D., Garland, R., Teanby, N. A., Fletcher, L. N., Orton, G. S., & Bézard, B. (2019). Probable detection of hydrogen sulphide (H2S) in Neptune’s atmosphere. Icarus, 321, 550-563. https://doi.org/10.1016/j.icarus.2018.12.014
  
  - Alday, J., Wilson, C. F., Irwin, P. G. J., Olsen, K. S., Baggio, L., Montmessin, F., et al. (2019). Oxygen isotopic ratios in Martian water vapour observed by ACS MIR on board the ExoMars Trace Gas Orbiter. Astronomy & Astrophysics, 630, A91. https://doi.org/10.1051/0004-6361/201936234
  
  - Irwin, P. G., Toledo, D., Garland, R., Teanby, N. A., Fletcher, L. N., Orton, G. A., & Bézard, B. (2018). Detection of hydrogen sulfide above the clouds in Uranus’s atmosphere. Nature Astronomy, 2(5), 420-427. https://doi.org/10.1038/s41550-018-0432-1

  - Irwin, P., Teanby, N., de Kok, R., Fletcher, L., Howett, C., Tsang, C., et al. (2008). The NEMESIS planetary atmosphere radiative transfer and retrieval tool. Journal of Quantitative Spectroscopy and Radiative Transfer, 109(6), 1136–1150. https://doi.org/10.1016/j.jqsrt.2007.11.006
  
.. toctree::
   :maxdepth: 2
 

.. toctree::
   :caption: Radiative transfer
   :hidden:
   
   radiative_transfer.ipynb

.. toctree::
   :caption: Retrievals
   :hidden:
   
   retrievals
   
.. toctree::
   :caption: Examples
   :hidden:
   
   examples
