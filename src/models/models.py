from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class BaseModel:
    """
    Base class for all models.
    Each child class must implement build().
    """

    def __init__(self, params=None):
        self.params = params or {}
        self.model = None

    def build(self):
        """Must be overridden by child classes"""
        raise NotImplementedError("Child classes must implement build()")

    def get_model(self):
        """Return the underlying sklearn model."""
        return self.model


# --------------------------------------------------------
# Logistic Regression
# --------------------------------------------------------
class LogisticRegressionModel(BaseModel):

    def build(self):
        self.model = LogisticRegression(**self.params)
        return self.model


# --------------------------------------------------------
# Random Forest
# --------------------------------------------------------
class RandomForestModel(BaseModel):

    def build(self):
        self.model = RandomForestClassifier(**self.params)
        return self.model


# --------------------------------------------------------
# XGBoost
# --------------------------------------------------------
class XGBoostModel(BaseModel):

    def build(self):
        self.model = XGBClassifier(
            **self.params,
            eval_metric="logloss",
            use_label_encoder=False
        )
        return self.model

class LightGBMModel(BaseModel):

    def build(self):
        self.model = LGBMClassifier(**self.params)
        return self.model