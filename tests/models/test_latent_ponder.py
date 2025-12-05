import torch
import torch
from omegaconf import OmegaConf

from airtrace.models.latent_ponder import LatentPonderWrapper
from airtrace.models.registry import build_model
from airtrace.tasks.one_step import OneStepTask


def _make_batch(batch_size: int = 3, seq_len: int = 6, dim: int = 4):
    torch.manual_seed(0)
    x = torch.randn(batch_size, seq_len, dim)
    # Target horizon of 1 for one-step task
    y = torch.randn(batch_size, 1, dim)
    return {"x": x, "y": y}


def test_latent_ponder_shapes_and_extras():
    torch.manual_seed(1)
    model = LatentPonderWrapper(
        input_dim=4,
        output_dim=2,
        max_steps=3,
        min_steps=1,
        ponder_penalty=0.05,
    )
    batch = _make_batch(dim=4)

    output = model(batch["x"])

    assert output["preds"].shape == (batch["x"].shape[0], 1, 2)
    extras = output["extras"]
    assert "halt_distribution" in extras
    assert extras["halt_distribution"].shape[1] == 3
    assert torch.isfinite(extras["ponder_loss"]).all()
    assert torch.all(extras["ponder_steps"] >= 1)


def test_halting_bias_controls_depth():
    torch.manual_seed(2)
    inputs = _make_batch(seq_len=5, dim=3)["x"]

    slow_model = LatentPonderWrapper(
        input_dim=3,
        output_dim=3,
        max_steps=4,
        min_steps=1,
        halt_bias=-10.0,
        ponder_penalty=0.0,
    )
    slow_model.eval()
    slow_steps = slow_model(inputs)["extras"]["ponder_steps"]

    fast_model = LatentPonderWrapper(
        input_dim=3,
        output_dim=3,
        max_steps=4,
        min_steps=1,
        halt_bias=10.0,
        ponder_penalty=0.0,
    )
    fast_model.eval()
    fast_steps = fast_model(inputs)["extras"]["ponder_steps"]

    assert slow_steps.max().item() == 4
    assert fast_steps.max().item() <= 2
    assert fast_steps.float().mean() < slow_steps.float().mean()


def test_trm_mode_refines_predictions_and_tracks_halting():
    torch.manual_seed(4)
    inputs = _make_batch(batch_size=2, seq_len=6, dim=3)["x"]

    model = LatentPonderWrapper(
        input_dim=3,
        output_dim=2,
        base_model={"name": "gru_ar", "params": {"dropout": 0.0, "hidden_size": 16}},
        trm_mode=True,
        max_steps=2,
        min_steps=2,
        halt_bias=-10.0,
        ponder_penalty=0.0,
        refine_head="mlp",
    )
    model.eval()

    base_only = model.base_model(inputs)
    output = model(inputs)

    assert output["preds"].shape == base_only["preds"].shape
    # TRM refinement should change the draft relative to the initial base prediction
    assert not torch.allclose(output["preds"], base_only["preds"])
    assert output["extras"]["halt_distribution"].shape[1] == 2
    assert torch.all(output["extras"]["ponder_steps"] >= 2)


def test_latent_ponder_builds_from_config():
    cfg = OmegaConf.load("src/airtrace/configs/model/latent_ponder.yaml")
    model = build_model(cfg.model, input_dim=3, output_dim=2)
    data = torch.randn(2, 8, 3)

    output = model(data)

    assert output["preds"].shape[0] == data.shape[0]
    assert "ponder_cost" in output["extras"]
    assert output["extras"]["halt_distribution"].shape[1] == cfg.model.params.max_steps


def test_task_applies_ponder_penalty():
    torch.manual_seed(3)
    model = LatentPonderWrapper(
        input_dim=4,
        output_dim=4,
        max_steps=2,
        min_steps=1,
        ponder_penalty=0.25,
    )
    task = OneStepTask({"loss": "mse", "metrics": ["rmse", "mae"]})
    batch = _make_batch(batch_size=2, seq_len=4, dim=4)

    result = task.training_step(batch, model)

    assert "ponder_cost" in result
    assert "ponder_steps" in result
    assert result["loss"].requires_grad
    assert result["ponder_cost"] >= 0


def test_trm_mode_uses_halting_loss_in_training():
    torch.manual_seed(7)
    model = LatentPonderWrapper(
        input_dim=3,
        output_dim=3,
        trm_mode=True,
        halting_mode="trm",
        max_steps=2,
        min_steps=1,
        halt_bias=0.0,
        dropout=0.0,
    )
    task = OneStepTask({"loss": "mse", "metrics": []})
    batch = _make_batch(batch_size=2, seq_len=4, dim=3)

    result = task.training_step(batch, model)

    assert "halting_loss" in result
    assert "ponder_loss" not in result
    assert result["loss"].requires_grad


def test_pondernet_replaces_task_loss_with_expected_objective():
    torch.manual_seed(8)
    model = LatentPonderWrapper(
        input_dim=2,
        output_dim=2,
        halting_mode="pondernet",
        max_steps=3,
        min_steps=1,
        ponder_penalty=0.1,
        dropout=0.0,
    )
    task = OneStepTask({"loss": "mae", "metrics": []})
    batch = _make_batch(batch_size=2, seq_len=3, dim=2)

    result = task.training_step(batch, model)

    assert "pondernet_loss" in result
    assert abs(result["loss"].item() - result["pondernet_loss"]) < 1e-5
    assert "ponder_loss" not in result


def test_supervision_steps_respect_aux_selection():
    torch.manual_seed(5)
    model = LatentPonderWrapper(
        input_dim=2,
        output_dim=2,
        max_steps=3,
        min_steps=1,
        aux_weight=0.5,
        supervision_steps=[1, -1],
    )
    inputs = torch.randn(2, 4, 2)

    output = model(inputs)

    aux_preds = output["extras"].get("aux_preds")
    assert aux_preds is not None
    # We expect one early supervision snapshot and one final snapshot
    assert aux_preds.shape[1] == 2
    assert aux_preds.shape[-1] == 2
