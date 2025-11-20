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


class TestPlotTimeseries:
    """Tests for plot_timeseries function."""

    def test_plot_timeseries_single_sensor(self):
        """Test plotting single sensor timeseries."""
        data = np.random.randn(100, 1)
        fig = plot_timeseries(data)
        
        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_plot_timeseries_multiple_sensors(self):
        """Test plotting multiple sensor timeseries."""
        data = np.random.randn(100, 3)
        sensor_names = ["fuel_flow", "mach", "altitude"]
        fig = plot_timeseries(data, sensor_names=sensor_names)
        
        assert fig is not None
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_plot_timeseries_custom_title(self):
        """Test custom title."""
        data = np.random.randn(50, 2)
        fig = plot_timeseries(data, title="Custom Title")
        
        assert "Custom Title" in fig._suptitle.get_text()
        plt.close(fig)

    def test_plot_timeseries_custom_figsize(self):
        """Test custom figure size."""
        data = np.random.randn(50, 2)
        fig = plot_timeseries(data, figsize=(8, 4))
        
        assert fig.get_figwidth() == 8
        assert fig.get_figheight() == 4
        plt.close(fig)


class TestPlotPredictions:
    """Tests for plot_predictions function."""

    def test_plot_predictions_basic(self):
        """Test basic prediction plotting."""
        targets = np.random.randn(100, 2)
        predictions = targets + np.random.randn(100, 2) * 0.1
        
        fig = plot_predictions(targets, predictions)
        
        assert fig is not None
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_plot_predictions_with_sensor_names(self):
        """Test prediction plotting with sensor names."""
        targets = np.random.randn(50, 3)
        predictions = targets + np.random.randn(50, 3) * 0.1
        sensor_names = ["fuel_flow", "mach", "altitude"]
        
        fig = plot_predictions(targets, predictions, sensor_names=sensor_names)
        
        assert fig is not None
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_plot_predictions_single_sensor(self):
        """Test prediction plotting for single sensor."""
        targets = np.random.randn(100, 1)
        predictions = targets + np.random.randn(100, 1) * 0.1
        
        fig = plot_predictions(targets, predictions)
        
        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)


class TestPlotErrorDistribution:
    """Tests for plot_error_distribution function."""

    def test_plot_error_distribution_basic(self):
        """Test basic error distribution plotting."""
        errors = np.random.randn(1000, 2)
        
        fig = plot_error_distribution(errors)
        
        assert fig is not None
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_plot_error_distribution_with_names(self):
        """Test error distribution with sensor names."""
        errors = np.random.randn(1000, 3)
        sensor_names = ["fuel_flow", "mach", "altitude"]
        
        fig = plot_error_distribution(errors, sensor_names=sensor_names)
        
        assert fig is not None
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_plot_error_distribution_single_sensor(self):
        """Test error distribution for single sensor."""
        errors = np.random.randn(1000, 1)
        
        fig = plot_error_distribution(errors)
        
        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)


class TestPlotScatterPredictions:
    """Tests for plot_scatter_predictions function."""

    def test_plot_scatter_predictions_basic(self):
        """Test basic scatter plot."""
        targets = np.random.randn(500, 2)
        predictions = targets + np.random.randn(500, 2) * 0.2
        
        fig = plot_scatter_predictions(targets, predictions)
        
        assert fig is not None
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_plot_scatter_predictions_with_names(self):
        """Test scatter plot with sensor names."""
        targets = np.random.randn(500, 3)
        predictions = targets + np.random.randn(500, 3) * 0.2
        sensor_names = ["fuel_flow", "mach", "altitude"]
        
        fig = plot_scatter_predictions(targets, predictions, sensor_names=sensor_names)
        
        assert fig is not None
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_plot_scatter_predictions_single_sensor(self):
        """Test scatter plot for single sensor."""
        targets = np.random.randn(500, 1)
        predictions = targets + np.random.randn(500, 1) * 0.2
        
        fig = plot_scatter_predictions(targets, predictions)
        
        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_plot_scatter_predictions_perfect(self):
        """Test scatter plot with perfect predictions."""
        targets = np.random.randn(500, 2)
        predictions = targets.copy()
        
        fig = plot_scatter_predictions(targets, predictions)
        
        assert fig is not None
        # Points should lie on diagonal
        plt.close(fig)


class TestPlotMetricsComparison:
    """Tests for plot_metrics_comparison function."""

    def test_plot_metrics_comparison_basic(self):
        """Test basic metrics comparison plot."""
        metrics_dict = {
            "model_a": {"mse": 0.1, "mae": 0.2, "rmse": 0.3},
            "model_b": {"mse": 0.15, "mae": 0.25, "rmse": 0.35},
            "model_c": {"mse": 0.12, "mae": 0.22, "rmse": 0.32},
        }
        
        fig = plot_metrics_comparison(metrics_dict)
        
        assert fig is not None
        assert len(fig.axes) == 3  # One for each metric
        plt.close(fig)

    def test_plot_metrics_comparison_single_metric(self):
        """Test metrics comparison with single metric."""
        metrics_dict = {
            "model_a": {"mse": 0.1},
            "model_b": {"mse": 0.15},
        }
        
        fig = plot_metrics_comparison(metrics_dict)
        
        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_plot_metrics_comparison_custom_title(self):
        """Test metrics comparison with custom title."""
        metrics_dict = {
            "model_a": {"mse": 0.1, "mae": 0.2},
            "model_b": {"mse": 0.15, "mae": 0.25},
        }
        
        fig = plot_metrics_comparison(metrics_dict, title="Custom Comparison")
        
        assert "Custom Comparison" in fig._suptitle.get_text()
        plt.close(fig)

    def test_plot_metrics_comparison_many_models(self):
        """Test metrics comparison with many models."""
        metrics_dict = {
            f"model_{i}": {"mse": 0.1 + i * 0.01, "mae": 0.2 + i * 0.01}
            for i in range(10)
        }
        
        fig = plot_metrics_comparison(metrics_dict, figsize=(15, 6))
        
        assert fig is not None
        assert len(fig.axes) == 2
        plt.close(fig)
