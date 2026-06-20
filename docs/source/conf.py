# -*- coding: utf-8 -*-
# Python Template sphinx config

# Import relevant modules
import sys
import os
import re
from importlib_metadata import metadata

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../../scripts"))

# -- General configuration ------------------------------------------------

# General information about the project.
project = "sp_validation"

mdata = metadata(project)
# PEP 621 stores contributors in 'Author-email' ("Name <email>, ...") rather
# than the legacy 'Author' field; strip the addresses for a clean author list.
author = mdata.get("Author") or re.sub(
    r"\s*<[^>]*>", "", mdata.get("Author-email", "CosmoStat")
)
version = mdata["Version"]
copyright = "2021, {}".format(author)

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = "3.3"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "myst_parser",
    "numpydoc",
]

# Include module names for objects
add_module_names = False

# Set class documentation standard.
autoclass_content = "class"

# Order docstrings as in the source
autodoc_member_order = "bysource"

# Include private class methods
autodoc_default_options = {"members": True, "private-members": True}

# Generate summaries
autosummary_generate = True

# Suppress class members in toctree.
numpydoc_show_class_members = False

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
show_authors = True

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "default"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinxawesome_theme"
# html_theme = 'sphinx_book_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "show_prev_next": True,
    "show_scrolltop": True,
    "globaltoc_includehidden": True,
}


# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "{0} v{1}".format(project, version)

# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = None

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = "%d %b, %Y"

# Convert quotes and dashes to typographically correct entities.
smartquotes = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# -- Intersphinx Mapping ----------------------------------------------

# Refer to the package libraries for type definitions
intersphinx_mapping = {
    "python": ("http://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("https://matplotlib.org", None),
}

# -- BibTeX Setting  ----------------------------------------------

bibtex_bibfiles = ["refs.bib"]
