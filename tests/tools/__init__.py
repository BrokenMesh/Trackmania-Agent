"""Tests of the tools/ scripts.

pytest (prepend import mode, no tests/__init__.py) imports this directory as the top-level
package `tools`, which would shadow the repository's tools/ package. Extending __path__
makes `import tools.render_replays` etc. resolve to the real scripts either way.
"""

from pathlib import Path

__path__.append(str(Path(__file__).resolve().parents[2] / "tools"))  # noqa: F821
