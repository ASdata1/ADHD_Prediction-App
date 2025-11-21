from src.data.data_loader import DataLoader
from src.preprocess.preprocessing import ADHDPreprocessor
from src.models.models import LogisticRegressionModel
from src.models.train import ModelTrainer
from src.models.hyperparameter_tuning import HyperparameterTuner
from src.models.evaluation import ModelEvaluator
from src.utils.saved_artifacts import save_model, save_metrics


class TrainPipeline:
    """
    Full ADHD ML training pipeline:
      1. Load & merge data
      2. Split raw data (train/val/test)
      3. Preprocess (scale → impute → encode → PCA → corr → ADASYN)
      4. Train baseline model
      5. Tune hyperparameters
      6. Retrain tuned model
      7. Evaluate & save results
    """

    def __init__(self, config):
        self.config = config

    def run(self):

        # -------------------------------------------------------------
        # 1. LOAD + MERGE RAW DATA
        # -------------------------------------------------------------
        loader = DataLoader(self.config)
        df = loader.merge_all()
        if df is None:
            raise RuntimeError("Failed to merge datasets. Pipeline stopped.")

        # -------------------------------------------------------------
        # 2. SPLIT RAW DATASET (NO LEAKAGE)
        # -------------------------------------------------------------
        pre = ADHDPreprocessor(
            quantitative_cols=self.config.quantitative_cols,
            categorical_cols=self.config.categorical_cols,
            pca_cols=self.config.pca_cols,
            pca_components=self.config.pca_components,
            corr_threshold=self.config.corr_threshold,
            random_state=self.config.random_state,
        )

        X_train, X_val, X_test, y_train, y_val, y_test = pre.split_raw_data(
            df=df,
            test_size=self.config.test_size,
            val_size=self.config.val_size,
            random_state=self.config.random_state
        )

        # -------------------------------------------------------------
        # 3. FULL PREPROCESSING PIPELINE
        # -------------------------------------------------------------
        X_train_bal, X_val_p, X_test_p, y_train_bal = pre.process(
            X_train, X_val, X_test, y_train
        )

        # -------------------------------------------------------------
        # 4. BASELINE MODEL TRAINING
        # -------------------------------------------------------------
        trainer = ModelTrainer(
            LogisticRegressionModel,
            self.config.logistic_regression_params
        )
        baseline_model = trainer.train(X_train_bal, y_train_bal)

        # -------------------------------------------------------------
        # 5. BASELINE VALIDATION EVALUATION
        # -------------------------------------------------------------
        evaluator = ModelEvaluator(self.config.results_path)
        baseline_metrics = evaluator.evaluate(baseline_model, X_val_p, y_val)
        print("\n BASELINE METRICS\n", baseline_metrics)

        # -------------------------------------------------------------
        # 6. HYPERPARAMETER TUNING (BAYESIAN)
        # -------------------------------------------------------------
        tuner = HyperparameterTuner(
            LogisticRegressionModel,
            param_space=self.config.lr_search_space
        )

        best_model, best_params = tuner.tune(X_train_bal, y_train_bal)
        print("\n🏆 BEST TUNED PARAMS:", best_params)

        # -------------------------------------------------------------
        # 7. RETRAIN FINAL MODEL WITH BEST PARAMS
        # -------------------------------------------------------------
        final_trainer = ModelTrainer(LogisticRegressionModel, best_params)
        tuned_model = final_trainer.train(X_train_bal, y_train_bal)

        tuned_metrics = evaluator.evaluate(tuned_model, X_val_p, y_val)
        print("\n🚀 TUNED METRICS\n", tuned_metrics)

        # -------------------------------------------------------------
        # 8. SAVE ARTIFACTS
        # -------------------------------------------------------------
        save_model(tuned_model, f"{self.config.models_path}/lr_tuned.joblib")
        save_metrics(tuned_metrics, f"{self.config.results_path}/lr_tuned.json")

        print("\n💾 Model + metrics saved successfully.")

        return tuned_metrics, tuned_model, baseline_metrics

    def __str__(self):
        return f"TrainPipeline(config={self.config})"
