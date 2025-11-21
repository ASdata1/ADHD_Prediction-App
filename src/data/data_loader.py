import pandas as pd
import os
from src.pipeline.config import Config

class DataLoader:
    """
    Helper class to load and merge multiple datasets safely using try/except.

    Parameters
    ----------
    main_path : str
        Path to the main dataset (behavioural + demographic data).
    cat_path : str
        Path to the categorical dataset (if separate).
    conn_path : str
        Path to the connectivity / neuroimaging dataset.
    solutions_path : str, optional
        Path to the solutions/labels file for supervised learning.
    key : str
        Column used to merge all datasets (e.g. subject_id).
    """


    def __init__(self, config):
        self.quant_path = config.quantitative_path
        self.cat_path = config.categorical_path
        self.conn_path = config.connectome_path
        self.solutions_path = config.solutions_path

        # Your merge key used in all datasets
        self.key = config.merge_key   
    # ------------------------------------------------------
    # Internal safe CSV loader
    # ------------------------------------------------------
    def _load_file(self, path):
        """Safe loader for both CSV and Excel files."""
        try:
            if not os.path.exists(path):
                print(f"[ERROR] File does not exist: {path}")
                return None

            if path.endswith(".xlsx") or path.endswith(".xls"):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)

            print(f"[OK] Loaded: {path}   Shape={df.shape}")
            return df

        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {e}")
            return None

    # ------------------------------------------------------
    # Public loaders
    # ------------------------------------------------------
    def load_quant(self): return self._load_file(self.quant_path)
    def load_cat(self): return self._load_file(self.cat_path)
    def load_connectome(self): return self._load_file(self.conn_path)
    def load_solutions(self): return self._load_file(self.solutions_path)


    # ------------------------------------------------------
    # Merge all datasets
    # ------------------------------------------------------
    def merge_all(self):
        df_quant = self.load_quant()
        df_cat = self.load_cat()
        df_conn = self.load_connectome()
        df_solutions = self.load_solutions()

        for name, df in {
            "quant": df_quant,
            "cat": df_cat,
            "conn": df_conn,
            "solutions": df_solutions,
        }.items():
            if df is None:
                raise RuntimeError(f"{name} dataset failed to load.")

            if self.key not in df.columns:
                raise RuntimeError(
                    f"Merge key '{self.key}' NOT found in {name} dataset.\n"
                    f"Columns in {name}: {list(df.columns)}"
                )

        df = df_quant.merge(df_cat, on=self.key, how="left")
        df = df.merge(df_conn, on=self.key, how="left")
        df = df.merge(df_solutions, on=self.key, how="left")

        print("[OK] Successfully merged all datasets. Final shape:", df.shape)
        return df
