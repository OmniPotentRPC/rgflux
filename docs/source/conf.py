"""Sphinx configuration for the rgflux project site."""

from __future__ import annotations

project = "rgflux"
copyright = "2026, Rohit Goswami"
author = "Rohit Goswami"
release = "0.2.0"
version = "0.2.0"

extensions = [
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "shibuya"
html_static_path = ["_static"]
html_favicon = "_static/favicon.svg"
html_logo = "_static/mark.svg"
html_title = "rgflux"
html_css_files = ["custom.css"]

html_context = {
    "source_type": "github",
    "source_user": "OmniPotentRPC",
    "source_repo": "rgflux",
    "source_version": "main",
    "source_docs_path": "/docs/orgmode/",
    "page_source_suffix": ".org",
}

html_theme_options = {
    "accent_color": "gold",
    "color_mode": "dark",
    "dark_code": True,
    "show_ai_links": False,
    "github_url": "https://github.com/OmniPotentRPC/rgflux",
    "nav_links": [
        {"title": "Get started", "url": "getting-started"},
        {"title": "How-to", "url": "howto"},
        {"title": "Reference", "url": "reference"},
        {"title": "Explanation", "url": "explanation"},
        {"title": "Source", "url": "https://github.com/OmniPotentRPC/rgflux/tree/develop/src", "external": True},
        {"title": "Search", "url": "search"},
    ],
}

intersphinx_mapping: dict = {}

# Publisher landing pages behind doi.org redirects (ACS, AIP, APS, Wiley, OUP,
# Royal Society) return HTTP 403 to automated linkcheck clients; ci/check_links.py
# verifies every DOI directly against the Handle System REST API.
linkcheck_ignore = [
    r"^https://doi\.org/",
]

copybutton_prompt_text = r"\$ "
copybutton_prompt_is_regexp = True

bibtex_bibfiles = ["references.bib"]
