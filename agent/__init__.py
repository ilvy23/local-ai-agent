import sys as _sys

if _sys.platform == "win32":                       # legacy Windows consoles / pipes: UTF-8 first (see agent/osport.py)
    from agent import osport as _osport

    _osport.windows_utf8_setup()
