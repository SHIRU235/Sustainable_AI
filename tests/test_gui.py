import unittest
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



from src.gui import app

class TestGUI(unittest.TestCase):

    def test_app_imports_without_crash(self):
        try:
            __import__("src.gui.app")
        except Exception as e:
            self.fail(f"Importing app.py crashed: {e}")

if __name__ == "__main__":
    unittest.main()