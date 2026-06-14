"""trail-scope backend package — qualitative single-image inference demo.

`main.py` stays thin (routes/validation/CORS/lifespan/errors); the real logic lives
here in `config`, `inference`, `preprocess`, `artifacts`, and the frozen `vendored`
inference core.
"""

from . import config

__version__ = config.VERSION
