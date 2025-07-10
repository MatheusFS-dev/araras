"""
Keras model summary utilities.

Functions:
    - capture_model_summary: Capture model summary as a string.

Example:
    >>> from araras.keras.utils.summary import capture_model_summary
    >>> capture_model_summary(...)
"""
from araras.commons import *

def capture_model_summary(model):
    """
    Capture model summary as a string.

    Args:
        model: Keras model

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
