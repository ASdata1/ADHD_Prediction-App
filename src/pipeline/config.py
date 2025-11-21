from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Config:

    """
    Configuration for the ADHD ML Pipeline.
    Everything here maps directly to the pipeline steps.
    """

    # ============================================================
    # GENERAL SETTINGS
    # ============================================================
    random_state: int = 42
    n_jobs: int = -1

    # ============================================================
    # DATA PATHS
    # ============================================================
    quantitative_path: str = r"C:\Users\04ama\Downloads\archive (27)\TRAIN_NEW (2)\TRAIN_NEW\TRAIN_QUANTITATIVE_METADATA_new.xlsx"
    categorical_path: str = r"C:\Users\04ama\Downloads\archive (27)\TRAIN_NEW (2)\TRAIN_NEW\TRAIN_CATEGORICAL_METADATA_new.xlsx"
    connectome_path: str   = r"C:\Users\04ama\Downloads\archive (27)\TRAIN_NEW (2)\TRAIN_NEW\TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv"
    solutions_path: str    = r"C:\Users\04ama\Downloads\archive (27)\TRAIN_NEW (2)\TRAIN_NEW\TRAINING_SOLUTIONS.xlsx"
    merge_key: str = "participant_id"


    # Where to save models + preprocessors + results
    models_path: str = r"C:\Users\04ama\OneDrive\chemistry\ADHD_SEX_Prediction\models"
    preprocessors_path: str = r"C:\Users\04ama\OneDrive\chemistry\ADHD_SEX_Prediction\preprocessors"
    results_path: str = r"C:\Users\04ama\OneDrive\chemistry\ADHD_SEX_Prediction\results"

    # ============================================================
    # DATA SPLITTING
    # ============================================================
    test_size: float = 0.20
    val_size: float = 0.25  # 0.25 of the remaining 80%

    # ============================================================
    # PREPROCESSING SETTINGS (from your ADHDPreprocessor)
    # ============================================================
    quantitative_cols: List[str] = None
    categorical_cols: List[str] = None
    pca_cols: List[str] = None
    pca_components: int = 10
    corr_threshold: float = 0.7

    # ============================================================
    # CROSS-VALIDATION
    # ============================================================
    cv_folds: int = 5
    scoring_metric: str = "f1_macro"

    # ============================================================
    # HYPERPARAMETER TUNING
    # ============================================================
    tuning_method: str = "bayesian"  # 'bayesian' is correct for you
    n_iter_bayes: int = 50

    # Bayesian search spaces
    lr_search_space: Dict[str, Any] = None
    rf_search_space: Dict[str, Any] = None

    # ============================================================
    # INITIAL BASELINE MODEL PARAMS
    # ============================================================
    logistic_regression_params: Dict[str, Any] = None
    random_forest_params: Dict[str, Any] = None

    # ============================================================
    # POST-INIT: FILL DEFAULTS
    # ============================================================
    def __post_init__(self):

        # Baseline LR params
        if self.logistic_regression_params is None:
            self.logistic_regression_params = {
                "max_iter": 200,
                "solver": "liblinear",
                "C": 1.0,
                "penalty": "l2"
            }

        # Baseline RF params
        if self.random_forest_params is None:
            self.random_forest_params = {
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt"
            }

        # Bayesian Search Spaces
        if self.lr_search_space is None:
            self.lr_search_space = {
                "C": (1e-4, 1e2, "log-uniform"),
                "max_iter": [200, 500, 1000],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "saga"]
            }

        if self.rf_search_space is None:
            self.rf_search_space = {
                "n_estimators": (50, 400),
                "max_depth": (3, 20),
                "min_samples_split": (2, 10),
                "min_samples_leaf": (1, 5),
                "max_features": ["sqrt", "log2"]
            }
