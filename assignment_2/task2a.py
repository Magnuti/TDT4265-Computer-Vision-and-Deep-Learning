import numpy as np
import utils
import typing
np.random.seed(1)


def pre_process_training_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"

    mean = X.mean()
    std = np.std(X)

    # Normalize
    X = (X - mean) / std

    # Bias trick
    X_with_bias = np.ones((X.shape[0], X.shape[1] + 1))
    X_with_bias[:, :-1] = X

    return X_with_bias, mean, std


def pre_process_non_training_images(X: np.ndarray, mean: float, std: float):
    """
    Helper function to normalize the validation/test data since we must
    normalize with the mean and standard validation from the training data.
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
        mean: average value of the training data
        stf: standard deviation of the training data
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"

    # Normalize
    X = (X - mean) / std

    # Bias trick
    X_with_bias = np.ones((X.shape[0], X.shape[1] + 1))
    X_with_bias[:, :-1] = X

    return X_with_bias


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"

    loss = - np.sum(targets * np.log(outputs), axis=1)
    return np.average(loss)


def sigmoid(x: np.ndarray) -> np.ndarray:
    # Clip to avoid numpy overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    return sigmoid(x) * (1 - sigmoid(x))


def improved_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.7159 * np.tanh((2 / 3) * x)


def improved_sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    # return 1.14393 / ((np.cosh((2 / 3) * x)) ** 2)
    # Taken from Piazza as my formula above leads to numpy overflow
    return 1.7159 * (2 / 3) * (1 - np.tanh(2 * x / 3)**2)


class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 28 * 28 + 1
        self.use_improved_sigmoid = use_improved_sigmoid

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            if use_improved_weight_init:
                std = 1 / np.sqrt(prev)
                w = np.random.normal(0, std, w_shape)
            else:
                w = np.random.uniform(-1, 1, w_shape)
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """

        self.activations = []
        self.weighted_inputs = []
        for i, weights in enumerate(self.ws):
            z = np.matmul(X, weights)  # Shape (batch size, num_outputs)
            if i < len(self.ws) - 1:
                # Sigmoid on the hidden layers
                X = sigmoid(z)
            else:
                # SoftMax as last layer
                exp = np.exp(z)  # Shape (batch size, num_outputs)
                exp_sum_per_output = np.sum(
                    exp, axis=1)  # Shape (batch size, )
                exp_sum_per_output = exp_sum_per_output.reshape(
                    X.shape[0], 1)  # Shape (batch size, 1)
                X = exp / exp_sum_per_output

            self.weighted_inputs.append(z)
            self.activations.append(X)

        return X

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        self.zero_grad()

        batch_size = X.shape[0]

        error = - (targets - outputs)  # Initial error based on the target
        for i in range(len(self.grads) - 1, 0, -1):
            self.grads[i] = np.matmul(
                self.activations[i - 1].T, error) / batch_size

            if self.use_improved_sigmoid:
                activation_gradient = improved_sigmoid_derivative(
                    self.weighted_inputs[i - 1])
            else:
                activation_gradient = sigmoid_derivative(
                    self.weighted_inputs[i - 1])

            error = np.matmul(error, self.ws[i].T) * activation_gradient

        self.grads[0] = np.matmul(X.T, error) / batch_size

        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    output = np.zeros((Y.shape[0], num_classes))
    numbers = Y[:, 0]
    indexes = np.arange(Y.shape[0])
    output[indexes, numbers] = 1
    return output


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train, mean, std = pre_process_training_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
