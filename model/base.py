class model:
    def __init__(self):
        self.timestamp = None
        self.args = None
    def train(self, data_train, data_valid=None, **kwargs) -> None:
        pass

    def predict(self, input_data, **kwargs):
        pass

    def load(self, model_path: str, **kwargs) -> None:
        pass
