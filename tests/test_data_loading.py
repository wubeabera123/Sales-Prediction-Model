import unittest
import pandas as pd
from unittest.mock import patch, mock_open
from io import StringIO
from scripts.load_data import Load_Data  # Replace with the correct import for your module

class TestLoadData(unittest.TestCase):

    @patch("pandas.read_csv")
    def test_load_data_success(self, mock_read_csv):
        """Test loading a valid CSV file."""
        # Simulate pandas read_csv returning a DataFrame
        mock_df = pd.DataFrame({"col1": [1, 3], "col2": [2, 4]})
        mock_read_csv.return_value = mock_df

        loader = Load_Data("test.csv")
        loader.load_data()

        # Assert that pandas read_csv was called once with the file path
        mock_read_csv.assert_called_once_with("test.csv")

        # Check if the data attribute contains the mock DataFrame
        self.assertIsNotNone(loader.data)
        pd.testing.assert_frame_equal(loader.data, mock_df)

    @patch("pandas.read_csv")
    def test_load_data_file_not_found(self, mock_read_csv):
        """Test file not found error."""
        mock_read_csv.side_effect = FileNotFoundError()

        loader = Load_Data("nonexistent.csv")
        with patch("builtins.print") as mocked_print:
            loader.load_data()
            mocked_print.assert_called_with("Error: File nonexistent.csv not found.")
    
    @patch("pandas.read_csv")
    def test_load_data_empty_file(self, mock_read_csv):
        """Test empty file error."""
        mock_read_csv.side_effect = pd.errors.EmptyDataError()

        loader = Load_Data("empty.csv")
        with patch("builtins.print") as mocked_print:
            loader.load_data()
            mocked_print.assert_called_with("Error: No data in file empty.csv.")

    @patch("pandas.read_csv")
    def test_load_data_parse_error(self, mock_read_csv):
        """Test parsing error."""
        mock_read_csv.side_effect = pd.errors.ParserError()

        loader = Load_Data("corrupt.csv")
        with patch("builtins.print") as mocked_print:
            loader.load_data()
            mocked_print.assert_called_with("Error: Error parsing data in file corrupt.csv.")

    def test_get_data_before_loading(self):
        """Test get_data before loading any data."""
        loader = Load_Data("test.csv")
        with patch("builtins.print") as mocked_print:
            data = loader.get_data()
            self.assertIsNone(data)
            mocked_print.assert_called_with("Data has not been loaded yet. Please call load_data first.")

    @patch("pandas.read_csv")
    def test_get_data_after_loading(self, mock_read_csv):
        """Test get_data after loading data."""
        mock_df = pd.DataFrame({"col1": [1, 3], "col2": [2, 4]})
        mock_read_csv.return_value = mock_df

        loader = Load_Data("test.csv")
        loader.load_data()

        data = loader.get_data()
        pd.testing.assert_frame_equal(data, mock_df)

if __name__ == "__main__":
    unittest.main()
