class Scheduler:
    def __init__(self):
        pass

    def schedule(self, latency:float):
        raise NotImplementedError("Scheduler.schedule() must be implemented by subclasses.")

    def set_insertion_weight(self, weight:float):
        raise NotImplementedError("Scheduler.set_insertion_weight() must be implemented by subclasses.")

    def set_update_weight(self, weight:float):
        raise NotImplementedError("Scheduler.set_update_weight() must be implemented by subclasses.")

    def set_embedding_weight(self, weight:float):
        raise NotImplementedError("Scheduler.set_embedding_weight() must be implemented by subclasses.")

class DefaultScheduler(Scheduler):
    def __init__(self, weights: list[float] | None = None, reserved: float = 0.1):
        super().__init__()
        if weights is None:
            weights = [1.0, 1.0, 1.0]

        if len(weights) != 3:
            raise ValueError("Weights must be a list of three floats.")

        self.weights = list(weights)
        self.reserved = reserved

    def schedule(self, latency:float):
        total_weight = sum(self.weights) + self.reserved

        return [latency * (w / total_weight) for w in self.weights]

    def set_insertion_weight(self, weight:float):
        self.weights[0] = weight

    def set_update_weight(self, weight:float):
        self.weights[1] = weight

    def set_embedding_weight(self, weight:float):
        self.weights[2] = weight






