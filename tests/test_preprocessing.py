import pandas as pd
from src.data.preprocessing import remove_exact_duplicates

def test_remove_exact_duplicates():
    df = pd.DataFrame({"x": [1, 1, 2]})
    assert len(remove_exact_duplicates(df)) == 2
