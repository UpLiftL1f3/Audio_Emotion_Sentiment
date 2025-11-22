"""
Model package initializer.

Intentionally left minimal. Individual models are imported dynamically
via `importlib.import_module("models.<impl_module>")` in `registry.py`,
so we do not eagerly import any specific model classes here.
"""

__all__ = []
