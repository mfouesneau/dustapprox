import unittest
import sys
sys.path.append('../')


class TestQuick(unittest.TestCase):

    def test_dependencies(self):
        """Check that the package dependencies are installed."""
        from dustapprox import __VERSION__
        print("code version: ", __VERSION__)


if __name__ == '__main__':
    unittest.main()