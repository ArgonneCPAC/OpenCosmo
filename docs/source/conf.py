# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "OpenCosmo"
copyright = "2025, OpenCosmo Team"
author = "OpenCosmo Team"
release = "0.7.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx_rtd_theme",
    "sphinxcontrib.autodoc_pydantic",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_title = "OpenCosmo Documentation"

# -- Options for autodoc -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
autodoc_typehints = "description"
autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_field_list_validators = False

autosectionlabel_maxdepth = 2

html_logo = "_static/opencosmo_icon_150x150_white.png"
html_favicon = "_static/opencosmo_icon_16x16.png"
