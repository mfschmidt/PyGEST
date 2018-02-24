""" context.py
Temporarily add the current project's root to the python path
so we can test THIS version of the project.

Each test file should include the following line at top:

    from .context import pygest

That way, THIS current project can be tested as written.

Thanks to Kenneth Reitz and python-guide.org for this.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pygest
