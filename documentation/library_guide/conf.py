#===============================================================================
# Copyright Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'oneAPI Libraries Documentation'
copyright = 'Intel Corporation'
author = 'Intel'

# The full version, including alpha/beta/rc tags
release = '1.0'

rst_epilog = """
.. include:: /variables.txt
"""

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx-prompt', 
    'sphinx_substitution_extensions'
	]

docbundle_settings = {
    'csv_dir':'csv_dir'
}

# The master toctree document.
master_doc = 'index'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]
html_logo = '_static/oneAPI-rgb-rev-100.png'
html_favicon = '_static/favicons.png'
#latex_logo = '_static/oneAPI-rgb-3000.png'
html_show_sourcelink = False
#html_js_files = ['custom.js']

# html_context = {
#     'css_files': [
#         '_static/style.css',  # override wide tables in RTD theme
#         ],
#     }

# html_theme = 'otc_tcs_sphinx_theme'
# html_theme_path = ['_themes']

# import sphinx_rtd_theme
html_theme = 'sphinx_book_theme'
#html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
#if on_rtd:
#    using_rtd_theme = True

# Theme options
html_theme_options = {
    # 'typekit_id': 'hiw1hhg',
    # 'analytics_id': '',
    # 'sticky_navigation': True,  # Set to False to disable the sticky nav while scrolling.
    #'logo_only': False,  # if we have a html_logo below, this shows /only/ the logo with no title text
    #'collapse_navigation': False,  # Collapse navigation (False makes it tree-like)
    # 'display_version': True,  # Display the docs version
    #'navigation_depth': 4  # Depth of the headers shown in the navigation bar
}

# -- DITA for AEM ------------------------------------------------------------

ditaxml_flat_map_to_title=True
ditaxml_make_flat=True

ditaxml_topic_meta={}
ditaxml_topic_meta["audience"]="guid:etm-7775758175444e1289d59d63457bb00d"
ditaxml_topic_meta["content type"]="User Guide"
ditaxml_topic_meta["description"]="For developers wanting to use the Intel® oneAPI DPC++ Library (oneDPL) Guide"
ditaxml_topic_meta["document title"]="Intel® oneAPI DPC++ Library (oneDPL) Guide"
ditaxml_topic_meta["download url"]=""
ditaxml_topic_meta["entitlement"]=""
ditaxml_topic_meta["entitlementtype"]=""
ditaxml_topic_meta["IDZ custom tags"]="guid:etm-86f5321aa04d4781ae1c3a9b1f8b8a49"
ditaxml_topic_meta["keywords"]="None"
ditaxml_topic_meta["language"]="en"
ditaxml_topic_meta["location"]="us"
ditaxml_topic_meta["menu"]="/content/data/globalelements/US/en/sub-navigation/idz/idz-oneAPI"
ditaxml_topic_meta["noindexfollowarchive"]="true"
ditaxml_topic_meta["operating system"]="guid:etm-d23b81f1319b4f0bb8ec859bcc84e2b9,guid:etm-cf0ee1fba3374ceea048ddac3e923cab"
ditaxml_topic_meta["primaryOwner"]="Stern, Alexandra M (lexi.stern@intel.com)"
ditaxml_topic_meta["programidentifier"]="idz"
ditaxml_topic_meta["programming language"]="guid:etm-e759606e77ad42549ba71c380d6d61e2"
ditaxml_topic_meta["published date"]="05/03/2021"
ditaxml_topic_meta["resourcetypeTag"]="guid:etm-15865f41343146919f486177b8dbb3f3"
ditaxml_topic_meta["secondary contenttype"]="guid:etm-74fc3401b6764c42ad8255f4feb9bd9e"
ditaxml_topic_meta["security classification"]="Public Content"
ditaxml_topic_meta["shortDescription"]="User guide for users of the Intel® oneAPI DPC++ Library (oneDPL)."
ditaxml_topic_meta["shortTitle"]="Intel® oneAPI DPC++ Library (oneDPL) Guide"
ditaxml_topic_meta["software"]="guid:etm-4c7a4593bba04ee2940ff6a1bc1bc95a,guid:etm-c307701b7daf4566a9fcefe4572de81f"
ditaxml_topic_meta["technology"]="guid:etm-6b088d69d83243a0aa3b986645a7e74b"

ditaxml_prod_info={}
ditaxml_prod_info["prodname"]="oneDPL"
ditaxml_prod_info["version"]="2021.3"

ditaxml_data_about={}
ditaxml_data_about["intelswd_aliasprefix"]={"datatype":"webAttr","value":"oneapi-dpcpp-library-guide"}