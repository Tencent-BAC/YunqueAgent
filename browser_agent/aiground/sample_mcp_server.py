# -*- coding: utf-8 -*-
"""
sample_mcp_server

Open-source sample entrypoint: starts browser_agent (prefixless HTTP + /mcp SSE).
We keep the legacy filename `sample_mcp_server.py` for convenience, but it delegates to `open_source_server`.
"""

import os
import sys

if __name__ == "__main__":  # noqa
    # Add project root to sys.path so `aiground` package can be imported.
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))  # noqa

from aiground.open_source_server import main


if __name__ == "__main__":
    main()
