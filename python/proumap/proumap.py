class ProUMAP:
    def __init__(self, config: Optional[ProUMAPConfig] = None):
        self.config = config or ProUMAPConfig()
        self.engine = ProgressiveEngine(self.config)

    def fit(self):
        pass

    def transform(self):
        pass

    def fit_transform(self):
        pass

    def get_embeddings(self):
        pass

    def get_progress(self):
        pass
