"""PDF Renderer — converts a Markdown string to a PDF file.

Uses the ``markdown`` library to render Markdown → HTML, then ``xhtml2pdf``
(which wraps ``reportlab``) to convert styled HTML → PDF.

Usage:
    from backend.reports.pdf_renderer import markdown_to_pdf
    markdown_to_pdf(md_string, "/path/to/output.pdf")
"""

from __future__ import annotations

import io

import markdown as _markdown
from xhtml2pdf import pisa

# ---------------------------------------------------------------------------
# CSS — professional compliance-report style
# ---------------------------------------------------------------------------

_CSS = """
@page {
    size: A4;
    margin: 2cm 2.2cm 2.5cm 2.2cm;
}

body {
    font-family: "Helvetica", "Arial", sans-serif;
    font-size: 10pt;
    line-height: 1.55;
    color: #1a1a2e;
}

h1 {
    font-size: 20pt;
    color: #0d1b2a;
    border-bottom: 3px solid #1565c0;
    padding-bottom: 6pt;
    margin-bottom: 14pt;
}

h2 {
    font-size: 14pt;
    color: #1565c0;
    margin-top: 20pt;
    margin-bottom: 6pt;
    border-bottom: 1px solid #cfd8dc;
    padding-bottom: 3pt;
}

h3 {
    font-size: 11pt;
    color: #37474f;
    margin-top: 14pt;
    margin-bottom: 4pt;
}

p {
    margin: 6pt 0;
}

ul, ol {
    margin: 4pt 0 4pt 18pt;
    padding: 0;
}

li {
    margin-bottom: 2pt;
}

strong {
    color: #1a1a2e;
}

em {
    color: #546e7a;
}

hr {
    border: none;
    border-top: 1px solid #cfd8dc;
    margin: 14pt 0;
}

blockquote {
    border-left: 3px solid #1565c0;
    margin: 8pt 0 8pt 12pt;
    padding: 4pt 8pt;
    background-color: #e3f2fd;
    color: #37474f;
    font-style: italic;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 10pt 0;
    font-size: 9pt;
}

th {
    background-color: #1565c0;
    color: white;
    padding: 5pt 7pt;
    text-align: left;
    font-weight: bold;
}

td {
    padding: 4pt 7pt;
    border-bottom: 1px solid #e0e0e0;
    vertical-align: top;
}

tr:nth-child(even) td {
    background-color: #f5f9ff;
}

code {
    font-family: "Courier New", monospace;
    font-size: 8.5pt;
    background-color: #eceff1;
    padding: 1pt 3pt;
    border-radius: 2pt;
}

pre {
    background-color: #eceff1;
    padding: 8pt;
    border-radius: 4pt;
    font-size: 8pt;
    white-space: pre-wrap;
    word-break: break-word;
}

/* Risk-level colour helpers used in text */
.critical { color: #b71c1c; font-weight: bold; }
.high     { color: #e65100; font-weight: bold; }
.medium   { color: #f57f17; font-weight: bold; }
.low      { color: #1b5e20; }
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def markdown_to_pdf(md_text: str, output_path: str) -> None:
    """Convert a Markdown string to a PDF file.

    Parameters
    ----------
    md_text : str
        Source Markdown content (may include tables, blockquotes, etc.).
    output_path : str
        Absolute path where the PDF should be written.
    """
    # Step 1 — Markdown → HTML
    html_body = _markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "nl2br"],
    )

    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>{_CSS}</style>
</head>
<body>
{html_body}
</body>
</html>"""

    # Step 2 — HTML → PDF via xhtml2pdf
    with open(output_path, "wb") as pdf_file:
        result = pisa.CreatePDF(
            io.StringIO(full_html),
            dest=pdf_file,
            encoding="utf-8",
        )

    if result.err:
        raise RuntimeError(
            f"xhtml2pdf reported {result.err} error(s) while rendering {output_path}"
        )
