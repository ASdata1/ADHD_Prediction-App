# src/pipeline/main.py

from src.pipeline.config import Config
from src.pipeline.train_pipeline import TrainPipeline
import pandas as pd
import os

def main():
    print("\n===================================================")
    print("        ADHD Machine Learning Pipeline")
    print("===================================================\n")

    # 1. Load configuration
    config = Config()
    print("Loaded config:")
    print(config)

    # 2. Run the training pipeline
    pipeline = TrainPipeline(config)
    tuned_metrics, baseline_metrics = pipeline.run()

    # 3. Show results
    print("\n===================== RESULTS =====================")
    print("Baseline Model Performance:")
    print(baseline_metrics)

    print("\nTuned Model Performance:")
    print(tuned_metrics)

    # 4. Save results in a clean table
    results_df = pd.DataFrame([
        {"Model": "Baseline Logistic Regression", **baseline_metrics},
        {"Model": "Tuned Logistic Regression", **tuned_metrics},
    ])

    results_path = os.path.join(config.results_path, "final_results_table.csv")
    results_df.to_csv(results_path, index=False)

    print("\nSaved final results table →", results_path)
    print("\n===================================================")
    print("                Pipeline Completed!!!!!")
    print("===================================================\n")



if __name__ == "__main__":
    main()
