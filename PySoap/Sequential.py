import numpy as np
import random

from PySoap.layers import Layer
from PySoap import get_error_function, get_metric_function
from PySoap.optimizers import Optimizer, get_optimizer


class Sequential:
    """ Build a neural network sequentially by using the .add function

        Attributes
        ----------
        layers : dict of int - class
            Store the individual layers as a form of classes

    """

    def __init__(self):
        """ Initialise a sequential class
        """
        self.layers = {}

        self.loss_function = None
        self.metric_function = None
        self.optimizer_bias = None
        self.optimizer_weights = None

    def add(self, layer):
        """ Add a new layer

            Notes
            -----
            The layers are stacked in the order that they are added in, hence the name sequential.

            Also, sequential models must always start with the `Input` class

            Parameters
            ----------
            layer : :obj:Layer
                A single layer in the network
        """

        if not isinstance(layer, Layer):
            raise TypeError('Only instances of Layer can be added to the sequential class.')

        self.layers[self.n + 1] = layer

    def predict(self, x_train, output_only=True):
        """ Forward propagate the input through the layers

            Parameters
            ----------
            x_train : np.array
                x_train is assumed to be a list of all the inputs to be forward propagated. In particular
                it is assumed that the 0th index of x_train is the index that inputs is accessed by

            output_only : :obj:`bool`, optional
                If set to true, then this function will return only the prediction of the neural
                network. If set to false then this will return the outputs of the individual
                layers. Unless back propagation is being performed, this should be set to true.

            Returns
            -------
            np.array
                The final layer output of the neural netowork

            OR (if `output_only = False`)

            dict of int - np.array, dict of int - np.array
                The first dictionary will store outputs of all layers before it is passed through the activation
                function.
                The second dictionary will store the outputs of all layers after it has passed through the
                activation function.
        """

        if output_only:
            working_z = x_train
            for i in range(1, self.n + 1):  # Propagate working_Z throughout the layers
                working_z = self.layers[i].predict(working_z, output_only=True)
            return working_z
        else:
            a_dict = {0: None}
            z_dict = {0: x_train}

            for i in range(1, self.n + 1):
                current_layer = self.layers[i]
                a_dict[i], z_dict[i] = current_layer.predict(z_dict[i - 1], output_only=False,
                                                             pre_activation_of_input=a_dict[i - 1])

                i += 1
            return a_dict, z_dict

    def evaluate(self, x_test, y_test):
        """ Return the MSE of the model prediction

            Parameters
            ----------
            x_test : np.array
                X_test is assumed to be a list of all the inputs to be forward propagated. In particular
                it is assumed that the first index of X_test is the index that inputs is accessed by
            y_test : np.array
                y_test is the associated list of outputs to the list of inputs X_test.

            Returns
            -------
            str
                The error
        """
        prediction = self.predict(x_test)

        loss_val = self.loss(prediction, y_test)

        eval_str = f'{self.loss_function}: {format(loss_val, ".4f")}'

        if self.metric_function is not None:
            metric_val = self.metric(prediction, y_test)
            eval_str += f' - {self.metric_function}: {format(metric_val, ".4f")}'

        return eval_str

    def train(self, x_train, y_train, epochs=100, batch_size=None, verbose=True):
        """ Train the neural network by means of back propagation

            Parameters
            ----------
            x_train : np.array
                x_train is assumed to be a list of all the inputs to be forward propagated. In particular
                it is assumed that the first index of x_train is the index that inputs is accessed by
            y_train : np.array
                y_train is the associated list of outputs to the list of inputs x_train. More specifically,
                the neural network will be trained to find the map x_train -> y_train

            epochs : :obj:`int`, optional
                Number of times the neural network will see the entire dataset
            batch_size : :obj:`int`, optional
                The batch size for gradient descent. If not defined then `batch_size` is set to the
                length of the dataset, i.e. traditional gradient descent.
            verbose : bool, optional
                If set to `True` then the model performance will be printed after each epoch

        """

        training_length = len(x_train)
        if batch_size is None:
            batch_size = training_length
        index = np.arange(training_length)

        for _ in range(epochs):
            if verbose:
                print(f'Training on {len(x_train)} samples')

            random.shuffle(index)
            for i in range(np.ceil(training_length / batch_size).astype(int)):
                start, end = i * batch_size, (i + 1) * batch_size
                batch_x, batch_y = x_train[index[start:end]], y_train[index[start:end]]

                self._back_prop(batch_x, batch_y)

            if verbose:
                start, end = 0, batch_size
                batch_x, batch_y = x_train[index[start:end]], y_train[index[start:end]]
                evaluation = self.evaluate(batch_x, batch_y)
                print(f'Epoch {_ + 1}/{epochs}')
                print(evaluation)

    def _back_prop(self, x_train, y_train):
        """ Perform a single back propagation of the batch `x_train`, `y_train`.

            Parameters
            ----------
            x_train : np.array
                x_train is assumed to be a list of all the inputs to be forward propagated. In particular
                it is assumed that the first index of x_train is the index that inputs is accessed by
            y_train : np.array
                y_train is the associated list of outputs to the list of inputs x_train. More specifically,
                the neural network will be trained to find the map x_train -> y_train
        """

        # Forward propagate
        a_dict, z_dict = self.predict(x_train, output_only=False)
        delta_dict, grad_dict = {}, {}

        # Compute delta for the last layer
        delta_dict[self.n] = self.loss(z_dict[self.n], y_train, grad=True)  # Gradient of output
        if self.loss_function == 'cross_entropy':
            delta_dict[self.n] = z_dict[self.n] - y_train

        # Compute the weight gradients for the i-th layer and then compute delta_{i-1} for the
        # next layer in the network
        for i in range(self.n, 2, -1):  # i = 1 is the Input class, no backpropagation needed
            grad_dict[i] = self.layers[i].get_weight_grad_(delta_dict[i], z_dict[i - 1])
            g_prime = self.layers[i - 1].activation_function_(a_dict[i - 1], grad=True)
            
            if type(self.layers[i]).__name__ in ['SoftChop', 'BatchNorm']:
                delta_dict[i - 1] = self.layers[i].get_delta_backprop_(g_prime, delta_dict[i], z_dict[i - 1])
            else:
                delta_dict[i - 1] = self.layers[i].get_delta_backprop_(g_prime, delta_dict[i])

        # Use Optimizer to find the gradient update
        bias_grads = self.optimizer_bias.step({key: val[0] for (key, val) in grad_dict.items()})
        weights_grads = self.optimizer_weights.step({key: val[1] for (key, val) in grad_dict.items()})

        # Update layer parameters
        for i in range(self.n, 2, -1):
            current_layer = self.layers[i]
            current_layer.update_parameters_(bias_grads[i], weights_grads[i])

    def summary(self):
        """ Prints a summary of the layers in the `sequential` class
        """

        def buffer_str(input_str, max_buffer=30):
            if len(input_str) < max_buffer:
                return input_str + ' ' * (max_buffer - len(input_str))
            return input_str

        max_str_len = max([len(layer.summary_()[0]) for layer in self.layers.values()])

        for layer in self.layers.values():
            layer_type = buffer_str(layer.summary_()[0], max_buffer=max_str_len + 3)
            layer_out = layer.summary_()[1]
            print(f'{layer_type} :    {layer_out}')

    def build(self, loss_function, optimizer, metrics=None):
        """ Once the model layers have been added this method must be called to initialise
            all the `sequential_layer` classes

            loss_function : str
                The loss function
            metrics : str, optional
                The name of the function to judge performance of the model
        """

        if isinstance(optimizer, Optimizer):
            self.optimizer_bias = optimizer
            self.optimizer_weights = self.optimizer_bias.new_instance()
        elif isinstance(optimizer, str):
            self.optimizer_bias = get_optimizer(optimizer)
            self.optimizer_weights = self.optimizer_bias.new_instance()
        else:
            raise ValueError("optimizer must be an instance of Optimizer or str")

        self.loss_function = loss_function
        self.metric_function = metrics

        previous_output_shape = None
        for i in range(1, self.n + 1):
            layer = self.layers[i]

            layer.build(previous_output_shape)
            previous_output_shape = layer.output_shape

    @property
    def loss(self):
        return get_error_function(self.loss_function)

    @property
    def metric(self):
        if self.metric_function is not None:
            return get_metric_function(self.metric_function)
        return None

    @staticmethod
    def simplify_key(working_dict):
        """ Given a recursive dictionary (a dictionary of dictionary of dictionary ...) this
            will simplify the dictionary so it is simply a dictionary of str - :obj: pair.

            Notes
            -----
            This should only be used in the method `save_method`
        """

        new_dict = {}
        for (key, val) in working_dict.items():
            if isinstance(val, dict):
                temp_dict = Sequential.simplify_key(val)
                new_dict.update({f'{key}/{new_key}': new_val for (new_key, new_val) in temp_dict.items()})
            else:
                new_dict.update({key: val})

        return new_dict

    @property
    def n(self):
        return len(self.layers)

    def __len__(self):
        return self.n

    def save_model(self, file_path):
        """ Save the model

            Parameters
            ----------
            file_path : str
                The file
        """

        # Save the model as a dictionary of dictionary
        layer_attributes = {key: layer.__dict__ for (key, layer) in self.layers.items()}
        layer_names = {key: type(layer).__name__ for (key, layer) in self.layers.items()}

        optimizer_attributes = {'optimizer_bias': self.optimizer_bias.__dict__,
                                'optimizer_weights': self.optimizer_weights.__dict__}
        optimizer_names = {'optimizer_bias': type(self.optimizer_bias).__name__,
                           'optimizer_weights': type(self.optimizer_weights).__name__}

        model_dict = {'layer_names': layer_names, 'layer_attributes': layer_attributes,
                      'optimizer_names': optimizer_names, 'optimizer_attributes': optimizer_attributes}

        # Simplify the dictionary of dictionary
        simplified_model_dict = self.simplify_key(model_dict)
        simplified_model_dict['loss_function'] = self.loss_function
        simplified_model_dict['metric_function'] = self.metric_function

        # hdf5 can't store None
        simplified_model_dict = {key: val for (key, val) in simplified_model_dict.items() if val is not None}

        import h5py

        with h5py.File(file_path, 'w') as h:
            for k, v in simplified_model_dict.items():
                h.create_dataset(k, data=v)
