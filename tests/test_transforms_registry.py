import numpy as np
import pytest

from airtrace.transforms.base import Compose, Transform
from airtrace.transforms.registry import TRANSFORM_REGISTRY, build_transforms, list_transforms, register


@pytest.fixture(autouse=True)
def reset_registry():
    """Ensure registry state is isolated across tests."""

    original = TRANSFORM_REGISTRY.copy()
    TRANSFORM_REGISTRY.clear()
    try:
        yield
    finally:
        TRANSFORM_REGISTRY.clear()
        TRANSFORM_REGISTRY.update(original)


def test_register_validates_uniqueness_and_type():
    class NotATransform:
        pass

    with pytest.raises(TypeError):
        register("bad")(NotATransform)

    @register("sample")
    class SampleTransform(Transform):
        def fit(self, dataset):
            return self

        def __call__(self, x, y, meta):
            return x, y, meta

        def inverse(self, x, y=None):
            return x, y

    with pytest.raises(ValueError):
        register("sample")(SampleTransform)


def test_build_transforms_composes_and_inverses_round_trip():
    @register("scale")
    class ScaleTransform(Transform):
        def __init__(self, factor: float = 1.0):
            super().__init__()
            self.factor = factor
            self.loaded_factor = None

        def fit(self, dataset):
            self.is_fitted = True
            return self

        def __call__(self, x, y, meta):
            updated_meta = {**meta, "factor": self.factor}
            return x * self.factor, y * self.factor, updated_meta

        def inverse(self, x, y=None):
            inv_y = None if y is None else y / self.factor
            return x / self.factor, inv_y

        def get_stats(self):
            return {"factor": self.factor}

        def set_stats(self, stats):
            self.loaded_factor = stats["factor"]

    compose = build_transforms(
        [
            {"name": "scale", "factor": 2.0},
            {"name": "scale", "factor": 0.5},
        ]
    )
    assert isinstance(compose, Compose)

    x = np.array([[1.0, 2.0]])
    y = np.array([[0.5, -1.0]])
    transformed_x, transformed_y, meta = compose(x, y, {"factor": None})

    np.testing.assert_allclose(transformed_x, x)  # 2.0 then 0.5 cancels
    np.testing.assert_allclose(transformed_y, y)
    assert meta["factor"] == 0.5

    inv_x, inv_y = compose.inverse(transformed_x, transformed_y)
    np.testing.assert_allclose(inv_x, x)
    np.testing.assert_allclose(inv_y, y)

    stats = compose.get_stats()
    assert set(stats.keys()) == {"ScaleTransform_0", "ScaleTransform_1"}

    compose.set_stats({"ScaleTransform_0": {"factor": 3.0}, "ScaleTransform_1": {"factor": 4.0}})
    assert compose.transforms[0].loaded_factor == 3.0
    assert compose.transforms[1].loaded_factor == 4.0


def test_build_transforms_validates_registered_names():
    @register("registered")
    class RegisteredTransform(Transform):
        def fit(self, dataset):
            return self

        def __call__(self, x, y, meta):
            return x, y, meta

        def inverse(self, x, y=None):
            return x, y

    error_message = "Transform 'missing' not found in registry."
    with pytest.raises(ValueError, match=error_message):
        build_transforms([{"name": "missing"}])

    available = list_transforms()
    assert available == sorted(available)
    assert "registered" in available
