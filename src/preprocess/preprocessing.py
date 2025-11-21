import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
from collections import Counter

class ADHDPreprocessor:
    """
    Clean preprocessing pipeline for ADHD dataset.

    Order:
    1. Scale quantitative columns only
    2. KNN impute (quant + cat)
    3. One-hot encode categorical columns
    4. PCA on connectome columns
    5. Remove correlated features
    6. ADASYN only on X_train
    """

    def __init__(
        self,
        quantitative_cols=None,
        categorical_cols=None,
        pca_cols=None,
        pca_components=None,
        corr_threshold=0.7,
        random_state=42
    ):
        self.quantitative_cols = quantitative_cols or []
        self.categorical_cols = categorical_cols or []
        self.pca_cols = pca_cols or []    # columns used for PCA
        self.pca_components = pca_components
        self.corr_threshold = corr_threshold
        self.random_state = random_state

        # Fitted transformers
        self.scaler = None
        self.imputer = None
        self.encoder = None
        self.pca = None
        self.removed_features = []

    # ----------------------------------------------------
    def __str__(self):
        return (
            "ADHDPreprocessor(\n"
            f"  quantitative_cols={self.quantitative_cols},\n"
            f"  categorical_cols={self.categorical_cols},\n"
            f"  pca_cols={self.pca_cols},\n"
            f"  pca_components={self.pca_components},\n"
            f"  corr_threshold={self.corr_threshold},\n"
            f"  random_state={self.random_state}\n"
            ")"
        )

    # ----------------------------------------------------
    # 1. SCALE (quant cols only)
    # ----------------------------------------------------
    def scale(self, df):
        """
        Scale only quantitative columns.
        """
        try:
            df = df.copy()

            self.scaler = StandardScaler()
            df[self.quantitative_cols] = self.scaler.fit_transform(
                df[self.quantitative_cols]
            )
            return df

        except Exception as e:
            print("[ERROR] Scaling failed:", e)
            return df

    def scale_transform(self, df):
        """
        Apply fitted scaler to val/test.
        """
        try:
            df = df.copy()
            df[self.quantitative_cols] = self.scaler.transform(df[self.quantitative_cols])
            return df

        except Exception as e:
            print("[ERROR] Scaling transform failed:", e)
            return df

    # ----------------------------------------------------
    # 2. KNN IMPUTATION (quant + cat)
    # ----------------------------------------------------
    def impute(self, df):
        """
        Apply KNN imputation to quantitative + categorical columns.
        """
        try:
            df = df.copy()

            impute_cols = self.quantitative_cols + self.categorical_cols

            self.imputer = KNNImputer(n_neighbors=5)
            df[impute_cols] = self.imputer.fit_transform(df[impute_cols])
            return df

        except Exception as e:
            print("[ERROR] Imputation failed:", e)
            return df

    def impute_transform(self, df):
        try:
            df = df.copy()
            impute_cols = self.quantitative_cols + self.categorical_cols
            df[impute_cols] = self.imputer.transform(df[impute_cols])
            return df
        except Exception as e:
            print("[ERROR] Imputation transform failed:", e)
            return df

    # ----------------------------------------------------
    # 3. ONE-HOT ENCODING
    # ----------------------------------------------------
    def encode(self, df):
        """
        Fit one-hot encoder on categorical columns.
        """
        try:
            df = df.copy()

            if not self.categorical_cols:
                return df

            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded = self.encoder.fit_transform(df[self.categorical_cols])

            encoded_df = pd.DataFrame(
                encoded,
                columns=self.encoder.get_feature_names_out(self.categorical_cols),
                index=df.index
            )

            df = df.drop(columns=self.categorical_cols)
            df = pd.concat([df, encoded_df], axis=1)
            return df

        except Exception as e:
            print("[ERROR] Encoding failed:", e)
            return df

    def encode_transform(self, df):
        """
        Apply fitted encoder to val/test.
        """
        try:
            df = df.copy()
            encoded = self.encoder.transform(df[self.categorical_cols])

            encoded_df = pd.DataFrame(
                encoded,
                columns=self.encoder.get_feature_names_out(self.categorical_cols),
                index=df.index
            )

            df = df.drop(columns=self.categorical_cols)
            df = pd.concat([df, encoded_df], axis=1)
            return df
 
        except Exception as e:
            print("[ERROR] Encoding transform failed:", e)
            return df

    # ----------------------------------------------------
    # 4. PCA (on connectome columns)
    # ----------------------------------------------------
    def apply_pca(self, X_train, X_val, X_test):
        """
        PCA ONLY on connectome columns.
        """
        try:
            self.pca = PCA(
                n_components=self.pca_components,
                random_state=self.random_state
            )

            X_train_pca = self.pca.fit_transform(X_train[self.pca_cols])
            X_val_pca = self.pca.transform(X_val[self.pca_cols])
            X_test_pca = self.pca.transform(X_test[self.pca_cols])

            # Rename columns
            pca_names = [f"conn_{i+1}" for i in range(self.pca_components)]

            return (
                pd.DataFrame(X_train_pca, columns=pca_names, index=X_train.index),
                pd.DataFrame(X_val_pca, columns=pca_names, index=X_val.index),
                pd.DataFrame(X_test_pca, columns=pca_names, index=X_test.index),
            )

        except Exception as e:
            print("[ERROR] PCA failed:", e)
            raise

    # ----------------------------------------------------
    # 5. Remove Correlated Features
    # ----------------------------------------------------
    def remove_correlated(self, df):
        """
        Remove correlated features after PCA + encoding.
        """
        try:
            corr = df.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))

            self.removed_features = [
                col for col in upper.columns if any(upper[col] > self.corr_threshold)
            ]

            return df.drop(columns=self.removed_features)

        except Exception as e:
            print("[ERROR] Correlation filtering failed:", e)
            return df

    # ----------------------------------------------------
    # 6. ADASYN (train only)
    # ----------------------------------------------------
    def balance(self, X, y):
        """
        ADASYN oversampling on TRAIN ONLY.
        """
        try:
            sampler = ADASYN(
                sampling_strategy="auto",
                n_neighbors=5,
                random_state=self.random_state
            )
            return sampler.fit_resample(X, y)

        except Exception as e:
            print("[ERROR] ADASYN failed:", e)
            raise

    

    # ----------------------------------------------------
    # SPLIT RAW MERGED DATASET
    # ----------------------------------------------------
    def split_raw_data(self, df, test_size=0.2, val_size=0.25, random_state=42):
        """
        Split the merged dataset into train/val/test while preventing data leakage.

        Parameters
        ----------
        df : pd.DataFrame
            Merged dataset containing features + 'ADHD_Outcome'
        test_size : float
            Proportion of data used for test split
        val_size : float
            Proportion of remaining train data used for validation
        random_state : int

        Returns
        -------
        X_train, X_val, X_test, y_train, y_val, y_test
        """

        print("\nSTEP 1: RAW DATA SPLITTING")

        # Drop target + ID from features
        X = df.drop(columns=['ADHD_Outcome', 'participant_id'], errors='ignore')
        y = df['ADHD_Outcome']

        print(f"   Dataset shape: {df.shape}")
        print(f"   Feature shape: {X.shape}")
        print(f"   Target distribution: {dict(Counter(y))}")

        # First split: Train+Val vs Test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )

        # Second split: Train vs Val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            stratify=y_temp,
            random_state=random_state
        )

        print("\n   Splits created successfully:")
        print(f"     TRAIN: {X_train.shape} | {dict(Counter(y_train))}")
        print(f"     VAL:   {X_val.shape}   | {dict(Counter(y_val))}")
        print(f"     TEST:  {X_test.shape}  | {dict(Counter(y_test))}")

        return X_train, X_val, X_test, y_train, y_val, y_test


        # ----------------------------------------------------
    # 7. FULL PIPELINE: TRAIN + VAL + TEST
    # ----------------------------------------------------
    def process(self, X_train, X_val, X_test, y_train):
        """
        Run the full preprocessing pipeline:

        1. Scale quantitative features (fit on train, transform val/test)
        2. KNN impute quantitative + categorical (fit on train, transform val/test)
        3. One-hot encode categoricals (fit on train, transform val/test)
        4. PCA on connectome columns (fit on train, transform val/test)
        5. Remove highly correlated features (fit on train, drop same from val/test)
        6. ADASYN balancing on the training set only

        Returns
        -------
        X_train_bal : pd.DataFrame
            Fully processed & balanced training features.
        X_val_clean : pd.DataFrame
            Fully processed validation features (no balancing).
        X_test_clean : pd.DataFrame
            Fully processed test features (no balancing).
        y_train_bal :
            Resampled training labels.
        """
        try:
            # 1. SCALE
            X_train_s = self.scale(X_train)
            X_val_s   = self.scale_transform(X_val)
            X_test_s  = self.scale_transform(X_test)

            # 2. IMPUTE
            X_train_i = self.impute(X_train_s)
            X_val_i   = self.impute_transform(X_val_s)
            X_test_i  = self.impute_transform(X_test_s)

            # 3. ENCODE
            X_train_e = self.encode(X_train_i)
            X_val_e   = self.encode_transform(X_val_i)
            X_test_e  = self.encode_transform(X_test_i)

            # 4. PCA on connectome columns
            X_train_pca, X_val_pca, X_test_pca = self.apply_pca(
                X_train_e, X_val_e, X_test_e
            )

            # Drop original PCA source cols and add new PCA features
            X_train_e = X_train_e.drop(columns=self.pca_cols)
            X_val_e   = X_val_e.drop(columns=self.pca_cols)
            X_test_e  = X_test_e.drop(columns=self.pca_cols)

            X_train_full = pd.concat([X_train_e, X_train_pca], axis=1)
            X_val_full   = pd.concat([X_val_e, X_val_pca], axis=1)
            X_test_full  = pd.concat([X_test_e, X_test_pca], axis=1)

            # 5. REMOVE CORRELATED FEATURES (fit on train)
            X_train_clean = self.remove_correlated(X_train_full)
            X_val_clean   = X_val_full.drop(columns=self.removed_features, errors="ignore")
            X_test_clean  = X_test_full.drop(columns=self.removed_features, errors="ignore")

            # 6. ADASYN BALANCING (TRAIN ONLY)
            X_train_bal, y_train_bal = self.balance(X_train_clean, y_train)

            return X_train_bal, X_val_clean, X_test_clean, y_train_bal

        except Exception as e:
            print("Preprocessing pipeline failed:", e)
            raise

    

