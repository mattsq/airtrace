import torch
from omegaconf import OmegaConf

from airtrace.models import build_model
from airtrace.models.residual_solver import ResidualSolver, ResidualSolverLoss


def _make_batch(batch_size: int = 3, seq_len: int = 6, dim: int = 4):
    torch.manual_seed(0)
    x = torch.randn(batch_size, seq_len, dim)
    y = torch.randn(batch_size, 1, dim)
    return {"x": x, "y": y}


def test_residual_solver_shapes_and_normalization():
    torch.manual_seed(1)
    model = ResidualSolver(input_dim=4, output_dim=2, max_steps=5)
    batch = _make_batch(dim=4)

    output = model(batch["x"])
    preds = output["preds"]
    extras = output["extras"]

    assert preds.shape == (batch["x"].shape[0], 1, 2)
    assert extras["halt_logits"].shape == (batch["x"].shape[0], 5)
    assert torch.allclose(extras["halt_distribution"].sum(dim=1), torch.ones(batch["x"].shape[0]))
    assert extras["step_preds"].shape == (batch["x"].shape[0], 5, 2)
    assert extras["residuals"].shape == (batch["x"].shape[0], 5, 2)
    assert torch.all(extras["expected_steps"] >= 1)


def test_halting_bonus_modulates_expected_steps():
    torch.manual_seed(2)
    inputs = _make_batch(seq_len=5, dim=3)["x"]

    fast_model = ResidualSolver(
        input_dim=3,
        output_dim=3,
        max_steps=6,
        residual_bonus_logit=8.0,
    )
    fast_model.eval()

    slow_model = ResidualSolver(
        input_dim=3,
        output_dim=3,
        max_steps=6,
        residual_bonus_logit=-8.0,
    )
    slow_model.eval()

    fast_steps = fast_model(inputs)["extras"]["expected_steps"]
    slow_steps = slow_model(inputs)["extras"]["expected_steps"]

    assert fast_steps.mean() < slow_steps.mean()
    assert slow_steps.max().item() >= 5


def test_inference_respects_threshold_and_handles_never_halt():
    torch.manual_seed(3)
    inputs = _make_batch(batch_size=2, seq_len=4, dim=2)["x"]

    eager_model = ResidualSolver(input_dim=2, output_dim=2, max_steps=4, residual_bonus_logit=9.0)
    eager_model.halting_head.bias.data.fill_(6.0)
    preds, steps = eager_model.inference(inputs, halt_threshold=0.6)

    assert preds.shape == (inputs.shape[0], 1, 2)
    assert torch.all(steps <= 2)

    stubborn_model = ResidualSolver(input_dim=2, output_dim=2, max_steps=4, residual_bonus_logit=-9.0)
    stubborn_model.halting_head.bias.data.fill_(-9.0)
    forward_extras = stubborn_model(inputs)["extras"]
    assert torch.allclose(forward_extras["halt_distribution"].sum(dim=1), torch.ones(inputs.shape[0]))

    preds_never, steps_never = stubborn_model.inference(inputs, halt_threshold=0.99)
    assert torch.all(steps_never == stubborn_model.config.max_steps)
    assert torch.isfinite(preds_never).all()


def test_loss_combines_components():
    torch.manual_seed(4)
    model = ResidualSolver(input_dim=3, output_dim=3, max_steps=3)
    loss_fn = ResidualSolverLoss(model.config)
    batch = _make_batch(batch_size=2, seq_len=5, dim=3)

    outputs = model(batch["x"])
    losses = loss_fn(outputs, batch["y"])

    assert set(losses.keys()) == {
        "total_loss",
        "task_loss",
        "weighted_step_loss",
        "consistency_loss",
        "compute_penalty",
        "halting_kl",
    }
    assert losses["total_loss"].requires_grad
    assert losses["task_loss"] <= losses["total_loss"]


def test_build_model_from_config():
    cfg = OmegaConf.load("src/airtrace/configs/model/residual_solver.yaml")
    model = build_model(cfg.model, input_dim=3, output_dim=2)
    assert isinstance(model, ResidualSolver)
    assert model.config.max_steps == cfg.model.params.max_steps
