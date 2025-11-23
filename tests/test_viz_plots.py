"""Tests for visualization utilities."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

# Use non-interactive backend for testing
matplotlib.use('Agg')

from airtrace.viz.plots import (
    plot_error_distribution,
    plot_metrics_comparison,
    plot_predictions,
    plot_scatter_predictions,
    plot_timeseries,
)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=0)


class TestPlotTimeseries:
    """Tests for plot_timeseries function."""

    def test_plot_timeseries_single_sensor(self, rng: np.random.Generator):
        """Timeseries plots should render the provided data and labels."""
        data = rng.standard_normal((100, 1))
        fig = plot_timeseries(data)

        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 1

        axis = fig.axes[0]
        lines = axis.get_lines()
        assert len(lines) == 1

        x_data, y_data = lines[0].get_data()
        assert len(x_data) == data.shape[0]
        np.testing.assert_allclose(y_data, data[:, 0])
        assert axis.get_ylabel() == "Sensor 0"
        assert fig.axes[-1].get_xlabel() == "Time"
        plt.close(fig)

    def test_plot_timeseries_multiple_sensors(self, rng: np.random.Generator):
        """Each sensor should render its own line with matching labels."""
        data = rng.standard_normal((120, 3))
        sensor_names = ["fuel_flow", "mach", "altitude"]
        fig = plot_timeseries(data, sensor_names=sensor_names)

        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 3

        for idx, (axis, expected_label) in enumerate(zip(fig.axes, sensor_names)):
            lines = axis.get_lines()
            assert len(lines) == 1
            _, y_data = lines[0].get_data()
            np.testing.assert_allclose(y_data, data[:, idx])
            assert axis.get_ylabel() == expected_label
        plt.close(fig)

    def test_plot_timeseries_custom_title(self, rng: np.random.Generator):
        """Custom titles should appear in the figure suptitle."""
        data = rng.standard_normal((50, 2))
        fig = plot_timeseries(data, title="Custom Title")

        assert "Custom Title" in fig._suptitle.get_text()
        plt.close(fig)

    def test_plot_timeseries_custom_figsize(self, rng: np.random.Generator):
        """Custom figsize should be applied to the figure."""
        data = rng.standard_normal((50, 2))
        fig = plot_timeseries(data, figsize=(8, 4))

        assert fig.get_figwidth() == 8
        assert fig.get_figheight() == 4
        plt.close(fig)


class TestPlotPredictions:
    """Tests for plot_predictions function."""

    def test_plot_predictions_basic(self, rng: np.random.Generator):
        """Predictions plot should overlay targets and predictions."""
        targets = rng.standard_normal((100, 2))
        predictions = targets + rng.standard_normal((100, 2)) * 0.1

        fig = plot_predictions(targets, predictions)

        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 2
        for idx, axis in enumerate(fig.axes):
            lines = axis.get_lines()
            assert {line.get_label() for line in lines} == {"Ground Truth", "Prediction"}
            assert len(lines) == 2
            gt_x, gt_y = lines[0].get_data()
            pred_x, pred_y = lines[1].get_data()
            assert len(gt_x) == targets.shape[0]
            np.testing.assert_allclose(gt_y, targets[:, idx])
            np.testing.assert_allclose(pred_y, predictions[:, idx])
        plt.close(fig)

    def test_plot_predictions_with_sensor_names(self, rng: np.random.Generator):
        """Sensor names should map to axis labels in prediction plots."""
        targets = rng.standard_normal((50, 3))
        predictions = targets + rng.standard_normal((50, 3)) * 0.1
        sensor_names = ["fuel_flow", "mach", "altitude"]

        fig = plot_predictions(targets, predictions, sensor_names=sensor_names)

        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 3
        for axis, expected_label in zip(fig.axes, sensor_names):
            assert axis.get_ylabel() == expected_label
        plt.close(fig)

    def test_plot_predictions_single_sensor(self, rng: np.random.Generator):
        """Single sensor predictions should still render both curves."""
        targets = rng.standard_normal((80, 1))
        predictions = targets + rng.standard_normal((80, 1)) * 0.05

        fig = plot_predictions(targets, predictions)

        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 1
        lines = fig.axes[0].get_lines()
        assert len(lines) == 2
        plt.close(fig)


class TestPlotErrorDistribution:
    """Tests for plot_error_distribution function."""

    def test_plot_error_distribution_basic(self, rng: np.random.Generator):
        """Histograms should contain counts for each sensor."""
        errors = rng.standard_normal((1000, 2))

        fig = plot_error_distribution(errors)

        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 2
        for axis in fig.axes:
            heights = [patch.get_height() for patch in axis.patches]
            assert heights, "Histogram should create bar patches"
            assert np.isclose(sum(heights), errors.shape[0])
            assert axis.get_xlabel() == "Error"
        plt.close(fig)

    def test_plot_error_distribution_with_names(self, rng: np.random.Generator):
        """Sensor names should appear in subplot titles."""
        errors = rng.standard_normal((500, 3))
        sensor_names = ["fuel_flow", "mach", "altitude"]

        fig = plot_error_distribution(errors, sensor_names=sensor_names)

        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 3
        titles = [axis.get_title() for axis in fig.axes]
        assert titles == sensor_names
        plt.close(fig)

    def test_plot_error_distribution_single_sensor(self, rng: np.random.Generator):
        """Single sensor histograms should retain shared y-axis labeling."""
        errors = rng.standard_normal((200, 1))

        fig = plot_error_distribution(errors)

        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 1
        assert fig.axes[0].get_ylabel() == "Frequency"
        plt.close(fig)


class TestPlotScatterPredictions:
    """Tests for plot_scatter_predictions function."""

    def test_plot_scatter_predictions_basic(self, rng: np.random.Generator):
        """Scatter plots should include all points and a diagonal guide."""
        targets = rng.standard_normal((500, 2))
        predictions = targets + rng.standard_normal((500, 2)) * 0.2

        fig = plot_scatter_predictions(targets, predictions)

        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 2
        for axis in fig.axes:
            scatter = axis.collections[0]
            offsets = scatter.get_offsets()
            assert offsets.shape[0] == targets.shape[0]
            assert len(axis.lines) >= 1
            diag_x, diag_y = axis.lines[0].get_data()
            np.testing.assert_allclose(diag_x, diag_y)
            assert axis.get_xlabel() == "Ground Truth"
            assert axis.get_ylabel() == "Prediction"
        plt.close(fig)

    def test_plot_scatter_predictions_with_names(self, rng: np.random.Generator):
        """Sensor names should populate subplot titles."""
        targets = rng.standard_normal((300, 3))
        predictions = targets + rng.standard_normal((300, 3)) * 0.2
        sensor_names = ["fuel_flow", "mach", "altitude"]

        fig = plot_scatter_predictions(targets, predictions, sensor_names=sensor_names)

        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 3
        assert [axis.get_title() for axis in fig.axes] == sensor_names
        plt.close(fig)

    def test_plot_scatter_predictions_single_sensor(self, rng: np.random.Generator):
        """Single sensor scatter plots still render the diagonal guide."""
        targets = rng.standard_normal((500, 1))
        predictions = targets + rng.standard_normal((500, 1)) * 0.2

        fig = plot_scatter_predictions(targets, predictions)

        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 1
        assert fig.axes[0].lines
        plt.close(fig)

    def test_plot_scatter_predictions_perfect(self, rng: np.random.Generator):
        """Perfect predictions should align exactly on the diagonal."""
        targets = rng.standard_normal((500, 2))
        predictions = targets.copy()

        fig = plot_scatter_predictions(targets, predictions)

        assert isinstance(fig, matplotlib.figure.Figure)
        for axis in fig.axes:
            scatter = axis.collections[0]
            offsets = scatter.get_offsets()
            np.testing.assert_allclose(offsets[:, 0], offsets[:, 1])
        plt.close(fig)


class TestPlotMetricsComparison:
    """Tests for plot_metrics_comparison function."""

    def test_plot_metrics_comparison_basic(self):
        """Metrics comparison should create bars per model and metric."""
        metrics_dict = {
            "model_a": {"mse": 0.1, "mae": 0.2, "rmse": 0.3},
            "model_b": {"mse": 0.15, "mae": 0.25, "rmse": 0.35},
            "model_c": {"mse": 0.12, "mae": 0.22, "rmse": 0.32},
        }

        fig = plot_metrics_comparison(metrics_dict)

        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 3  # One for each metric
        for metric_name, axis in zip(["mse", "mae", "rmse"], fig.axes):
            heights = [patch.get_height() for patch in axis.patches]
            assert len(heights) == len(metrics_dict)
            expected = [metrics_dict[model][metric_name] for model in metrics_dict]
            np.testing.assert_allclose(sorted(heights), sorted(expected))
            assert axis.get_title() == metric_name.upper()
        plt.close(fig)

    def test_plot_metrics_comparison_single_metric(self):
        """Single-metric comparisons should collapse to one subplot."""
        metrics_dict = {
            "model_a": {"mse": 0.1},
            "model_b": {"mse": 0.15},
        }

        fig = plot_metrics_comparison(metrics_dict)

        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 1
        heights = [patch.get_height() for patch in fig.axes[0].patches]
        np.testing.assert_allclose(sorted(heights), [0.1, 0.15])
        plt.close(fig)

    def test_plot_metrics_comparison_custom_title(self):
        """Custom titles should propagate to the figure."""
        metrics_dict = {
            "model_a": {"mse": 0.1, "mae": 0.2},
            "model_b": {"mse": 0.15, "mae": 0.25},
        }

        fig = plot_metrics_comparison(metrics_dict, title="Custom Comparison")

        assert "Custom Comparison" in fig._suptitle.get_text()
        plt.close(fig)

    def test_plot_metrics_comparison_many_models(self):
        """Larger model sets should still render all metrics."""
        metrics_dict = {
            f"model_{i}": {"mse": 0.1 + i * 0.01, "mae": 0.2 + i * 0.01}
            for i in range(10)
        }

        fig = plot_metrics_comparison(metrics_dict, figsize=(15, 6))

        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 2
        for axis in fig.axes:
            assert len(axis.patches) == len(metrics_dict)
        plt.close(fig)
