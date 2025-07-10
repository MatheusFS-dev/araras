"""
This module provides a function to build convolutional neural network (CNN) blocks.

Functions:
    - build_cnn1d: Builds a 1D convolutional layer with optional hyperparameter tuning and regularization.
    - build_dense_as_conv1d: Simulates a Dense layer using a Conv1D layer with specific configurations.
    - build_cnn2d: Builds a 2D convolutional layer with optional hyperparameter tuning and regularization.
    - build_dense_as_conv2d: Simulates a Dense layer using a Conv2D layer with specific configurations.
    - build_cnn3d: Builds a 3D convolutional layer with optional hyperparameter tuning and regularization.
    - build_dense_as_conv3d: Simulates a Dense layer using a Conv3D layer with specific configurations.

Usage example:
    from araras.keras.builders.cnn import build_cnn1d
    from araras.keras.hparams import HParams
    import tensorflow as tf
    hparams = HParams()
    x = tf.keras.Input(shape=(128, 64))  # Example input shape
    cnn_layer = build_cnn1d(
        trial=trial,
        hparams=hparams,
        x=x,
        filters_range=(32, 128),
        kernel_size_range=(3, 7),
        use_batch_norm=True
    )

    from araras.keras.builders.dnn import build_dnn
    hparams = HParams()
    x = tf.keras.Input(shape=(128, 64))  # Example input shape
    dnn_layer = build_dnn(
        trial=trial,
        hparams=hparams,
        x=x,
        units_range=(64, 256),
        units_step=32,
        dropout_rate_range=(0.1, 0.5),
        dropout_rate_step=0.1,
        use_batch_norm=True
    )
"""
