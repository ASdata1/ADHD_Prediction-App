class ModelTrainer:
    """
    Handles training of any model that inherits BaseModel.
    """

    def __init__(self, model_class, params):
        self.model_obj = model_class(params)
        self.model = self.model_obj.build()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model

    def get_model(self):
        return self.model
