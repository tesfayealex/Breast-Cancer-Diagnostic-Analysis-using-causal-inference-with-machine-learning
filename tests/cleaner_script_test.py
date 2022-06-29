

from cleaner import CleanDataFrame
import unittest
from bleach import Cleaner
import pandas as pd
import numpy as np
import sys
import os
from pprint import pprint
import json
# sys.path.append(os.path.abspath(os.path.join('../Pharmaceutical-Sales-prediction/scripts')))
sys.path.append('../scripts')


class TestCleanerModule(unittest.TestCase):

    def setUp(self) -> pd.DataFrame:
        self.cleaner = CleanDataFrame()

    def test_get_numerical_columns(self):
        df = pd.DataFrame([[1, "a", "b", 4], [5, "c", "d", 8]])
        self.assertEqual(self.cleaner.get_numerical_columns(df), [0, 3])

    def test_get_categorical_columns(self):
        df = pd.DataFrame([[1, "a", "b", 4], [5, "c", "d", 8]])
        self.assertEqual(self.cleaner.get_categorical_columns(df), [1, 2])

    def test_percent_missing(self):
        df = pd.DataFrame([[1, "a", "b", ], [5, "c", "d", 8]])
        self.assertEqual(self.cleaner.percent_missing(df), 12.5)

    def test_replace_value(self):
        df = pd.DataFrame([[1, "a", "b", ], [5, "c", "d", 8]])
        df.rename(columns={0: "first"},)
        print(df.rename(columns={0: "first"}))
        fullfilled_df = pd.DataFrame([[1, "a", "b"], [5, "c", "d", 8]])
        self.assertEqual(self.cleaner.replace_value(
            df, "first", 1), fullfilled_df)


if __name__ == '__main__':
    unittest.main()
