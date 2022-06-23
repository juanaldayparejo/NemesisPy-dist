project = 'NemesisPy'

extensions = [
    "nbsphinx",
    "sphinx_gallery.load_style",
]


#Defining each image shown in the gallery
nbsphinx_thumbnails = {
    'examples/mars_solocc/mars_SO': '_static/exomars_SO.jpg',
    'examples/Jupiter_CIRS_nadir_thermal_emission/Jupiter_CIRS': '_static/jupiter_cassini.jpg',
    'examples/Measurement/Measurement': '_static/observation_sketch.png',
    'examples/Stellar/StellarExample': '_static/solar_spec.jpg',
    'examples/Atmosphere/Atmosphere': '_static/planetary_atmospheres.png',
    'examples/MieScattering/MieScattering': '_static/mars_sunset2.jpg'
}


#Defining the actual appearance of the website
html_theme = 'sphinx_rtd_theme'


#For changing the style of the jupyter notebooks
nbsphinx_prolog = """
{% set docname = env.doc2path(env.docname, base=None) %}

.. only:: html

    Go there: https://example.org/notebooks/{{ docname }}

.. only:: latex

    The following section was created from :file:`{{ docname }}`.
"""
