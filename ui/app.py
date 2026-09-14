from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def index():
    """
    Codebase Cartography — AST Explorer + CFG/Call Graphs.

    The entire application (styles, D3 rendering, CFG/call-graph
    generation, JSON validation, drag-and-drop upload) lives inside
    templates/index.html. It runs client-side in the browser: the
    user drops or picks an analysis JSON file (or loads the built-in
    sample dataset) and everything is parsed and rendered locally.
    Flask's only job here is to serve that one page.
    """
    return render_template('index.html')


if __name__ == '__main__':
    # debug=True + host=0.0.0.0 exposes the Werkzeug debugger (remote code
    # execution) to anyone who can reach this host. Fine for local-only dev;
    # gate it behind an env var so it can't accidentally ship that way.
    import os
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)
