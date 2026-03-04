import unittest
import numpy as np

from star_shadow import timeseries_functions as tsf


class TestLikelihoodFunction(unittest.TestCase):
    def setUp(self):
        """Setup common test inputs before each test."""
        self.times = np.array([0, 1, 2, 3, 4])
        self.residuals = np.array([1.0, -0.5, 0.2, -0.1, 0.05])
        self.signal_err = np.array([0.1, 0.1, 0.1, 0.1, 0.1])  # Constant error

    def test_likelihood_without_errors(self):
        """Test that likelihood runs and returns a finite number."""
        likelihood = tsf.calc_likelihood_2(self.times, self.residuals, None)
        self.assertTrue(np.isfinite(likelihood), "Likelihood should be finite")

    def test_likelihood_with_errors(self):
        """Test that likelihood with measurement errors also returns a finite number."""
        likelihood = tsf.calc_likelihood_2(self.times, self.residuals, self.signal_err)
        self.assertTrue(np.isfinite(likelihood), "Likelihood with errors should be finite")

    def test_likelihood_deterministic_output(self):
        """Ensure function returns the same result for same input."""
        like1 = tsf.calc_likelihood_2(self.times, self.residuals, self.signal_err)
        like2 = tsf.calc_likelihood_2(self.times, self.residuals, self.signal_err)
        self.assertAlmostEqual(like1, like2, places=8, msg="Likelihood should be deterministic")

    def test_likelihood_changes_with_data(self):
        """Ensure likelihood changes when residuals change."""
        like_original = tsf.calc_likelihood_2(self.times, self.residuals, self.signal_err)
        modified_residuals = self.residuals + 0.1  # Slightly alter residuals
        like_modified = tsf.calc_likelihood_2(self.times, modified_residuals, self.signal_err)
        self.assertNotEqual(like_original, like_modified, "Likelihood should change when residuals change")


if __name__ == '__main__':
    unittest.main()
