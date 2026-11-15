"""Production entry point.

`main.py` doubles as a terminal chat client, so importing it would start the
interactive loop. Instead, production servers (gunicorn, Render, Railway,
Docker) import this module, which only exposes the Flask app.

Local development keeps working exactly as before:

    python main.py

Production:

    gunicorn wsgi:app --workers 1 --threads 8 --timeout 180
"""

from __future__ import annotations

from main import app

__all__ = ["app"]
