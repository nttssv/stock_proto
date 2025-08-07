from flask import Flask, render_template
import os
from pathlib import Path

app = Flask(__name__)

@app.route('/')
def index():
    static_folder = Path("static")
    files = sorted(
        [f for f in static_folder.iterdir() if f.is_file()],
        key=lambda f: f.stat().st_mtime,
        reverse=True
    )
    return render_template('index.html', files=files)

if __name__ == "__main__":
    app.run(debug=True)