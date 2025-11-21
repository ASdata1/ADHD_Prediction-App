try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
except ImportError:
    raise ImportError("scikit-optimize is required. Install with: pip install scikit-optimize")


class HyperparameterTuner:
    """
    Performs Bayesian optimization for any model class.
    """

    def __init__(self, model_class, search_space, cv=5, scoring="f1_macro"):
        self.model_class = model_class
        self.search_space = search_space
        self.cv = cv
        self.scoring = scoring

    def tune(self, X_train, y_train):

        model = self.model_class().build()

        opt = BayesSearchCV(
            estimator=model,
            search_spaces=self.search_space,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1
        )

        opt.fit(X_train, y_train)

        return opt.best_estimator_, opt.best_params_
