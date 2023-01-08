import sys
import os
sys.path.insert(0,os.path.abspath('../NemesisPy/Path/'))

project = 'NemesisPy'

extensions = [
    "nbsphinx",
    "sphinx_gallery.load_style",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
#    "sphinx.ext.viewcode",
    "IPython.sphinxext.ipython_console_highlighting",
    ]


#Defining paramters for autodoc documentation
#napoleon_google_docstring = False
#napoleon_numpy_docstring = True
#napoleon_include_init_with_doc = False
#napoleon_include_private_with_doc = True
#napoleon_include_special_with_doc = True
#napoleon_use_admonition_for_examples = True
#napoleon_use_admonition_for_notes = True
#napoleon_use_admonition_for_references = False
#napoleon_use_ivar = True
#napoleon_use_keyword = True
#napoleon_use_param = True
#napoleon_use_rtype = True
#napoleon_preprocess_types = False
#napoleon_type_aliases = None
#napoleon_attr_annotations = False


#Defining each image shown in the gallery
nbsphinx_thumbnails = {
    'examples/mars_solocc/mars_SO': '_static/exomars_SO.jpg',
    'examples/Jupiter_CIRS_nadir_thermal_emission/Jupiter_CIRS': '_static/jupiter_cassini.jpg',
    'examples/Measurement/Measurement': '_static/observation_sketch.png',
    'examples/Stellar/StellarExample': '_static/solar_spec.jpg',
    'examples/Atmosphere/Atmosphere': '_static/planetary_atmospheres.png',
    'examples/MieScattering/MieScattering': '_static/mars_sunset2.jpg',
    'examples/Surface/surface': '_static/mars_surface.png'
}

#Defining the actual appearance of the website
html_theme = 'sphinx_rtd_theme'


exclude_patterns = ['_build', '**.ipynb_checkpoints']
