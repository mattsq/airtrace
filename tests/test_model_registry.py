import pytest
import torch

from airtrace.models.base import ARBaseModel
from airtrace.models import registry


class _TinyModel(ARBaseModel):
    def forward(self, x: torch.Tensor, context=None, **kwargs):
        preds = torch.zeros(x.shape[0], x.shape[1], self.output_dim)
        return {"preds": preds}


@pytest.fixture()
def model_registry_restore():
    original = registry.MODEL_REGISTRY.copy()
    registry.MODEL_REGISTRY.clear()
    yield
    registry.MODEL_REGISTRY.clear()
    registry.MODEL_REGISTRY.update(original)


def test_register_and_build_model(model_registry_restore):
    @registry.register("tiny")
    class Tiny(_TinyModel):
        pass

    built = registry.build_model({"name": "tiny", "params": {}}, input_dim=4, output_dim=2)

    assert isinstance(built, Tiny)
    output = built(torch.ones(2, 3, 4))
    assert output["preds"].shape == (2, 3, 2)
    assert torch.all(output["preds"] == 0)
    assert registry.list_models() == ["tiny"]


def test_model_registry_validation(model_registry_restore):
    class NotAModel:
        pass

    with pytest.raises(TypeError):
        registry.register("invalid")(NotAModel)

    @registry.register("tiny")
    class Tiny(_TinyModel):
        pass

    with pytest.raises(ValueError):
        registry.register("tiny")(Tiny)

    with pytest.raises(ValueError) as exc:
        registry.build_model({"name": "missing"}, input_dim=1, output_dim=1)

    assert "Available models" in str(exc.value)
