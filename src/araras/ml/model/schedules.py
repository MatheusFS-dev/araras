from araras.core import *


from tensorflow.keras.utils import register_keras_serializable  # tf.keras
import tensorflow as tf


@register_keras_serializable(package="custom")
class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay.

    This schedule increases the learning rate linearly from zero to `base_lr`
    during the first `warmup_steps`, then decays it smoothly to near zero using
    a half cosine curve over the remaining steps.

    The schedule is continuous at the warmup boundary, which avoids spikes at
    the join. It is piecewise smooth. The warmup branch is linear, the decay
    branch is cosine. Numerical guards are added to avoid division by zero.

    Mathematical definition for step t:
      If t < W:
        lr(t) = base_lr * t / max(1, W)
      Else:
        lr(t) = 0.5 * base_lr * (1 + cos(pi * (t - W) / max(1, T - W)))

    Where W is warmup_steps and T is total_steps.

    Args:
      base_lr: Target learning rate reached at the end of warmup and used as
        the peak value before cosine decay starts. Can be float or scalar tensor.
      warmup_steps: Number of optimizer steps used for linear warmup.
      total_steps: Total number of optimizer steps for the full schedule.
      name: Optional name scope for the schedule.

    Attributes:
      base_lr: Float, stored as Python float for serialization.
      warmup_steps: Int, warmup duration in steps.
      total_steps: Int, total number of steps for the schedule.
      name: String name for the object.
      verbose: Int, verbosity level. Either 0 (silent) or 1 (verbose).

    Example:
      ```python
      steps_per_epoch = max(1, len(X_tr) // BATCH_SIZE)
      total_steps = steps_per_epoch * EPOCHS
      warmup_steps = max(10, int(0.05 * total_steps))

      lr_schedule = WarmupCosine(base_lr=3e-4,
                                 warmup_steps=warmup_steps,
                                 total_steps=total_steps)

      optimizer = tf.keras.optimizers.AdamW(
          learning_rate=lr_schedule,
          weight_decay=2e-4,
          clipnorm=1.0
      )
      ```
    """

    def __init__(self, base_lr, warmup_steps, total_steps, name="warmup_cosine", verbose=0):
        super().__init__()
        self.base_lr = float(base_lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.name = name
        # Force 0 or 1
        self.verbose = 1 if int(verbose) == 1 else 0

    def _lr_at(self, t):
        """Pure, vectorized lr computation for any float step t."""
        t = tf.cast(t, tf.float32)
        w = tf.cast(self.warmup_steps, tf.float32)
        denom = tf.maximum(1.0, tf.cast(self.total_steps - self.warmup_steps, tf.float32))
        base = tf.cast(self.base_lr, tf.float32)
        pi = tf.constant(3.141592653589793, dtype=tf.float32)

        warm = base * (t / tf.maximum(1.0, w))
        cos_arg = tf.clip_by_value(t - w, 0.0, denom)
        cos_decay = 0.5 * base * (1.0 + tf.cos(pi * cos_arg / denom))
        return tf.where(t < w, warm, cos_decay)

    def __call__(self, step):
        """Compute learning rate at a given global optimizer step.

        Args:
          step: Integer step index, scalar Tensor or Python int.

        Returns:
          Scalar Tensor, dtype float32, the learning rate for this step.
        """
        # Current lr
        t = tf.cast(step, tf.float32)
        lr = self._lr_at(t)

        if self.verbose == 1:
            # Previous-step lr, computed without keeping state
            t_prev = tf.maximum(t - 1.0, 0.0)
            lr_prev = self._lr_at(t_prev)
            d_lr = lr - lr_prev

            # Phase string
            phase = tf.where(t < tf.cast(self.warmup_steps, tf.float32), "warmup", "cosine")

            # Print once per step
            step_i = tf.cast(step, tf.int64)
            tf.print("[WarmupCosine]", "step:", step_i, "lr:", lr, "d_lr:", d_lr, "phase:", phase)

        return lr

    def get_config(self):
        """Return JSON-serializable config."""
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "name": self.name,
            "verbose": int(self.verbose),
        }

    @classmethod
    def from_config(cls, config):
        """Recreate an instance from a config created by get_config."""
        return cls(**config)
