import PySoap

from inspect import signature
import h5py


def unpack_recursive_hdf5(file):
    unpacked = {}
    for key in file.keys():
        if isinstance(file[key], h5py._hl.group.Group):
            unpacked.update({key: unpack_recursive_hdf5(file[key])})
        else:
            unpacked.update({key: file[key][()]})
    return unpacked


def unpack_sequential_model(file_path):
    """ Unpack the model details from `file_path`.

        Parameters
        ----------
        file_path : str
            The file path

        Notes
        -----
        It is assumed that `file_path` is a HDF5 file, stored in the format as given in
        the `ml_files.Sequential` class
    """

    with h5py.File(file_path, 'r') as h:
        # For a given layer, load their attributes
        unpacked_hdf5 = unpack_recursive_hdf5(h)

    layer_attributes = unpacked_hdf5['layer_attributes']
    layer_names = unpacked_hdf5['layer_names']
    loss_function = unpacked_hdf5['loss_function']
    if 'metric_function' in unpacked_hdf5:
        metric_function = unpacked_hdf5['metric_function']
    else:
        metric_function = None

    return layer_attributes, layer_names, loss_function, metric_function


def load_model(file_path):
    """ Load model from `file_path`

        Parameters
        ----------
        file_path : str
            File Path to model

        Notes
        -----
        It is assumed that `file_path` is a HDF5 file, stored in the format as given in
        the `ml_files.Sequential` class
    """

    with h5py.File(file_path, 'r') as h:
        # For a given layer, load their attributes
        unpacked_hdf5 = unpack_recursive_hdf5(h)

    # Create a new instance of the Sequential class
    rebuilt_model = ml_files.Sequential()
    rebuilt_model.loss_function = unpacked_hdf5['loss_function']
    rebuilt_model.metric_function = unpacked_hdf5['metric_function'] if 'metric_function' in unpacked_hdf5 else None

    # Add Optimizers
    optimizer_names, optimizer_attributes = unpacked_hdf5['optimizer_names'], unpacked_hdf5['optimizer_attributes']
    for optimizer_ in ['optimizer_bias', 'optimizer_weights']:
        # Create instance of optimizer
        rebuilt_model.__dict__[optimizer_] = ml_files.optimizers.__dict__[optimizer_names[optimizer_]]()

        # Update optimizer attributes
        rebuilt_model.__dict__[optimizer_].__dict__.update(optimizer_attributes[optimizer_])

    # Add Layers
    layer_attributes, layer_names = unpacked_hdf5['layer_attributes'], unpacked_hdf5['layer_names']
    num_layers = len(layer_names.keys())

    for i in range(1, num_layers + 1):
        name, attributes = layer_names[str(i)], layer_attributes[str(i)]

        layer_class = ml_files.layers.__dict__[name]  # Get the layer class object

        sig = signature(layer_class.__init__).parameters
        kwargs = {key: attributes[key] for key in sig.keys() if key in attributes}

        rebuilt_model.add(layer_class(**kwargs))
        rebuilt_model.layers[i].__dict__.update(attributes)  # Update attributes of layer instance

    return rebuilt_model
