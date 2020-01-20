class Config:

    def __init__(self, path: str = None, **kwargs):
        self.population_size = kwargs.get("population_size", None)

        self.c1 = kwargs.get("c1", None)
        self.c2 = kwargs.get("c2", None)
        self.c3 = kwargs.get("c3", None)
        self.t = kwargs.get("t", None)

        self.train_counter = kwargs.get("train_counter", None)
        self.train_max_iterations = kwargs.get("train_max_iterations", None)
        self.train_elitism = kwargs.get("train_elitism", None)
        self.train_f = kwargs.get("train_f", None)
        self.train_cr = kwargs.get("train_cr", None)

        self.input_size = kwargs.get("input_size", None)
        self.bias_size = kwargs.get("bias_size", None)
        self.output_size = kwargs.get("output_size", None)
