# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------

project = 'YOM Bundle Recommender System'
copyright = '2026, Diana Khutorna'
author = 'Diana Khutorna'
release = '1.0'
version = '1.0.0'

# The master toctree document.
master_doc = 'index'

# -- General configuration --------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'myst_parser',
]

# Source suffix
source_suffix = {
    '.rst': None,
    '.md': 'markdown',
}

# MyST configuration
myst_enable_extensions = [
    'colon_fence',
    'html_image',
    'linkify',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Pygments lexer for code highlighting
pygments_style = 'sphinx'

# Language
language = 'en'

# -- Options for HTML output -----------------------------------------------

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'analytics_id': '',  # Your GTM or GA tracking ID
    'analytics_anonymize_ip': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': '#2862a3',
    'collapse_navigation': True,
    'sticky_navigation': False,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
    'logo_only': False,
    'canonical_url': 'https://yom-recommender.readthedocs.io/',
}

html_logo = None  # Add logo if available
html_favicon = None

html_static_path = ['_static']
html_css_files = []

# HTML context
html_context = {
    'display_github': True,
    'github_user': 'dianakhutorna',
    'github_repo': 'bundle_recommender',
    'github_version': 'final',
    'conf_py_path': '/docs/source/',
}

# -- Options for HTMLHelp output -------------------------------------------

htmlhelp_basename = 'YOMRecommenderdoc'

# -- Options for LaTeX output -----------------------------------------------

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
}

latex_documents = [
    (master_doc, 'YOMRecommender.tex', 'YOM Recommender System Documentation',
     'Diana Khutorna', 'manual'),
]

# -- Options for manual page output -----------------------------------------

man_pages = [
    (master_doc, 'yom-recommender', 'YOM Recommender System Documentation',
     [author], 1)
]

# -- Options for Texinfo output ---------------------------------------------

texinfo_documents = [
    (master_doc, 'YOMRecommender', 'YOM Recommender System Documentation',
     author, 'YOMRecommender', 'One line description of project.',
     'Miscellaneous'),
]

# -- Options for Epub output ------------------------------------------------

epub_basename = 'YOMRecommender'
epub_author = author
epub_language = language
epub_publisher = author
epub_copyright = copyright

# -- Extension configuration -----------------------------------------------

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'lightgbm': ('https://lightgbm.readthedocs.io/en/latest/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_keyword = True
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'show-inheritance': True,
}

# Mathjax configuration
mathjax_inline = [r'\(', r'\)']
mathjax_display = [r'\[', r'\]']

# -- Options for ReadTheDocs -------------------------------------------

# This is also automatically detected, if you use RTD
on_rtd = os.environ.get('READTHEDOCS') == 'True'
