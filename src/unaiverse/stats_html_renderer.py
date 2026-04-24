"""Minimal HTML wrapper for the base Stats.plot() output.

Converts a Plotly-format JSON payload (``{"data": [...], "layout": {...}}``) into a
self-contained ``<!DOCTYPE html>`` document that renders the chart client-side using
Plotly.js loaded from CDN.  Python never imports plotly.
"""

_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
  <style>
    html, body {{
      margin: 0; padding: 0;
      background: #0E0F14;
      color: #C8CDD3;
      font-family: sans-serif;
    }}
    #plot {{
      width: 100vw;
      height: 100vh;
    }}
  </style>
</head>
<body>
  <div id="plot"></div>
  <script>
    var DASH = {dash_json};
    Plotly.newPlot('plot', DASH.data, DASH.layout, {{responsive: true}});
  </script>
</body>
</html>
"""


def render_plotly_html(dash_json: str, title: str = "World Overview") -> str:
    """Wrap a Plotly-format JSON string in a self-contained HTML document.

    Args:
        dash_json: JSON string with ``'data'`` and ``'layout'`` keys, as produced by
            ``DefaultBaseDash.to_json()`` or ``UIPlot.to_json()``.
        title: Browser tab title for the document.

    Returns:
        A ``<!DOCTYPE html>`` string that renders the chart via Plotly.js CDN.
    """
    return _TEMPLATE.format(title=title, dash_json=dash_json)
