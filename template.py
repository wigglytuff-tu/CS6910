
class AutoDiffFunction():
    """Format for any function in general which has to be auto-differentiable
    """

    def __init__(self, *args, **kwds) -> None:
        self.saved_for_backward = {} # caching to save computation
        self.grad = {} # holds gradients

    def __call__(self, *args, **kwds):
        
        # performs one forward and backward pass in each call
        output = self.forward(*args, **kwds)
        self.grad = self.compute_grad(*args, **kwds)
        return output

    def forward(self, *args, **kwds):
        """Calculates a forward pass
        """
        pass

    def compute_grad(self, *args, **kwds):
        """Computes local gradient of that function itself w.r.t its input
        """
        pass

    def backward(self, *args, **kwds):
        """Computes actual gradient w.r.t. the loss after chained gradients ahead 
            of the function till the loss are passed 
        """
        pass


class Layer(AutoDiffFunction):
    """Format to create your own custom layer for the model
    """
    def __init__(self, *args, **kwds) -> None:
        super().__init__(*args, **kwds)

        self.weights = {} # holds weights of the layer
        self.optimizer = None # optimizer for the layer

    def initialize_weights(self, *args, **kwds):
        """Initialize weights for the layer
        """
        pass

    def update_weights(self):
        """Updates weights of the layer using layer's assigned optimizer
        """
        self.optimizer.step(self)


class Loss(AutoDiffFunction):
    """Format to create a custom loss function
    """

    def forward(self, y_true, y_pred):
        """Calculates a forward pass
        """
        pass 

    def backward(self):
        """Computes actual gradient w.r.t. the loss after chained gradients ahead 
            of the function till the loss are passed 
        """
        return self.grad["x"]

    def compute_grad(self, y_true, y_pred):
        """Computes local gradient of that function itself w.r.t its input
        """
        pass


class Optimizer():
    """Format to create a custom optimizer
    """
    def __init__(self, *args, **kwds):
        self.remember = {} # remembering parameters from last iteration
        pass

    def step(self, layer):
        """Performs the update step for weights of the optmizer

        Args:
            layer: Layer object assigned to the optimizer 
        """
        pass   