"""
Run the full test suite with zero extra dependencies:

    python run_tests.py

Uses stdlib unittest discovery on purpose (no pytest) — this is meant to
run identically on a fresh i9 laptop with nothing installed but Python and
this repo's own code, and on whatever grading environment doesn't have
pytest preinstalled either.
"""
import sys
import unittest

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests", pattern="test_*.py", top_level_dir=".")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
