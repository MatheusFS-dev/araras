import sys
from pathlib import Path

import pytest

# Ensure package is discoverable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

tf = pytest.importorskip("tensorflow")

from araras.ml.model.callbacks import get_callbacks_model


def test_restore_best_weights_requires_checkpoint(tmp_path):
    with pytest.raises(ValueError):
        get_callbacks_model(
            checkpoint_dir=None,
            early_stopping_patience=None,
            restore_best_weights=True,
        )

    callbacks_list = get_callbacks_model(
        checkpoint_dir=str(tmp_path),
        early_stopping_patience=None,
        reduce_lr_patience=None,
        restore_best_weights=True,
    )
    names = [cb.__class__.__name__ for cb in callbacks_list]
    assert "RestoreBestWeights" in names

