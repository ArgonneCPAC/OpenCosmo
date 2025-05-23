# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "OpenCosmo"
copyright = "2025, OpenCosmo Team"
author = "OpenCosmo Team"
release = "0.6.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinxcontrib.autodoc_pydantic",
]

templates_path = ["_templates"]
exclude_patterns = []
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

# -- Options for autodoc -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
autodoc_typehints = "description"
autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_field_list_validators = False

html_logo = "_static/opencosmo_icon_150x150_white.png"
html_favicon = "_static/opencosmo_icon_16x16.png"
