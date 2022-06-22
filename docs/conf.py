project = 'NemesisPy'

extensions = [
    "nbsphinx",
    "sphinx_gallery.load_style",
]

nbsphinx_thumbnails = {
    'examples/mars_solocc/mars_SO': '_static/exomars_SO.jpg',
    'examples/Jupiter_CIRS_nadir_thermal_emission/Jupiter_CIRS': '_static/jupiter_cassini.jpg',
    'examples/Measurement/Measurement': 'examples/Measurement/observation_sketch.png'
}

html_theme = 'sphinx_rtd_theme'
