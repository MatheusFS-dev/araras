from typing import Optional, Tuple

import time
import warnings

import tensorflow as tf
from tensorflow.keras import backend as K

from araras.ml.common.device import parse_device_spec
from araras.utils.loading_bar import gen_loading_bar
from araras.utils.verbose_printer import VerbosePrinter


vp = VerbosePrinter()


def capture_model_summary(model):
    """Capture model summary as a string.

    Args:
        model (Any): Keras model

    Returns:
        str: Model summary as string
    """
    summary_list = []
    model.summary(
        print_fn=lambda x: summary_list.append(x),
        expand_nested=True,
        show_trainable=True,
    )
    return "\n".join(summary_list)


def run_dummy_inference(
    model: tf.keras.Model,
    batch_size: int = 1,
    device: str = "cpu",
    warmup_runs: Optional[int] = None,
    runs: int = 1,
    verbose: int = 1,
) -> Tuple[float, float]:
    """
    Execute dummy inference passes on ``model`` and time them.

    The helper creates zero-filled tensors matching ``model.inputs`` for the
    requested ``batch_size`` and runs the model repeatedly on the selected
    device. Optional warm-up executions may be performed before timing begins
    to exclude one-off initialisation overheads. Each measured run converts the
    outputs to NumPy arrays to synchronise the execution graph and capture the
    true latency.

    Args:
        model (tf.keras.Model): Model whose inference latency should be
            measured.
        batch_size (int): Batch size for the dummy inputs. Defaults to ``1``.
        device (str): Device specification. Accepts ``"cpu"`` or
            ``"gpu/<index>"``. ``"both"`` is not supported. Defaults to
            ``"cpu"``.
        warmup_runs (Optional[int]): Number of warm-up executions performed
            before timing. ``None`` disables warm-ups. Defaults to ``None``.
        runs (int): Number of timed executions. Must be positive. Defaults to
            ``1``.
        verbose (int): Verbosity level. Values greater than zero render a
            progress bar. Defaults to ``1``.

    Returns:
        Tuple[float, float]: Average and peak inference latency in seconds.

    Raises:
        ValueError: If ``runs`` is less than ``1``, if ``batch_size`` is less
            than ``1``, or if ``device`` resolves to ``"both"``.
        RuntimeError: If the requested GPU device is unavailable.

    Notes:
        The model is executed inside a TensorFlow device context matching the
        requested ``device``. When a GPU is selected TensorFlow must have a
        visible physical GPU with the given index.
    """

    if runs < 1:
        raise ValueError("runs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    device_kind, gpu_index = parse_device_spec(device)
    if device_kind == "both":
        raise ValueError("device must be either 'cpu' or 'gpu/<index>'")

    if device_kind == "gpu":
        if gpu_index is None:
            raise ValueError("GPU device index could not be resolved")
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus or gpu_index >= len(gpus):
            raise RuntimeError(f"No GPU found for index {gpu_index}")
        device_label = f"GPU:{gpu_index}"
        device_spec = f"/GPU:{gpu_index}"
    else:
        device_label = "CPU"
        device_spec = "/CPU:0"

    warmups = max(int(warmup_runs or 0), 0)

    shapes = [(batch_size,) + tuple(K.int_shape(inp)[1:]) for inp in model.inputs]
    dummy_inputs = [
        tf.zeros(shape, dtype=inp.dtype) for shape, inp in zip(shapes, model.inputs)
    ]

    @tf.function
    def _infer(*args):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The structure of `inputs`",
                category=UserWarning,
            )
            return model(list(args), training=False)

    def _execute_once() -> float:
        with tf.device(device_spec):
            start = time.perf_counter()
            outputs = _infer(*dummy_inputs)
            tf.nest.map_structure(lambda tensor: tensor.numpy(), outputs)
            return time.perf_counter() - start
        
    try:
        for _ in range(warmups):
            _execute_once()

        iterator = range(runs)
        if verbose > 0:
            iterator = gen_loading_bar(
                iterator,
                description=vp.color(f"Calculating latency on {device_label}", "blue"),
                total=runs,
                bar_color="blue",
            )

        measurements = [
            _execute_once()
            for _ in iterator
        ]
    except Exception as exc:
        vp.logf(f"Error during inference on {device_label}: {exc}", log_level="ERROR", tag=vp.gen_tag("fileline"), color="red")

    if not measurements:
        return 0.0, 0.0

    average = sum(measurements) / len(measurements)
    peak = max(measurements)
    return average, peak
