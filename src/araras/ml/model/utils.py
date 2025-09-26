from araras.core import *


def capture_model_summary(model):
    """
    Capture model summary as a string.

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
