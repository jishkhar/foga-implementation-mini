import unittest
from api import parse_optimizer_output

class TestParsing(unittest.TestCase):
    def test_foga_parsing(self):
        output = """
Gen   1 | Valid: 277/277 | Best: 0.12345678s | Avg: 0.23456789s
Gen   2 | Valid: 277/277 | Best: 0.11111111s | Avg: 0.22222222s
        """
        result = parse_optimizer_output("foga", output)
        self.assertIn("history", result)
        self.assertEqual(len(result["history"]), 2)
        self.assertEqual(result["history"][0]["iteration"], 1)
        self.assertEqual(result["history"][0]["best"], 0.12345678)
        self.assertEqual(result["history"][0]["avg"], 0.23456789)

    def test_hbrf_parsing(self):
        output = """
Sampling 1/100...
BO Iteration 1: 0.123s (Best: 0.111s)
Adding -falign-functions: 0.111s -> 0.100s
        """
        result = parse_optimizer_output("hbrf_optimizer", output)
        self.assertIn("history", result)
        # We expect BO iteration and Greedy addition
        self.assertEqual(len(result["history"]), 2)
        self.assertEqual(result["history"][0]["iteration"], 101)
        self.assertEqual(result["history"][0]["best"], 0.111)
        self.assertEqual(result["history"][1]["iteration"], "Greedy")
        self.assertEqual(result["history"][1]["best"], 0.100)

    def test_xgboost_parsing(self):
        output = """
Iteration 1/50: Best: 0.123s
Iteration 2/50: Best: 0.111s
        """
        result = parse_optimizer_output("xgboost_optimizer", output)
        self.assertIn("history", result)
        self.assertEqual(len(result["history"]), 2)
        self.assertEqual(result["history"][0]["iteration"], 101)
        self.assertEqual(result["history"][0]["best"], 0.123)

if __name__ == '__main__':
    unittest.main()
