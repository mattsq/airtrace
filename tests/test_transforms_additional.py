"""Additional tests for transforms to improve coverage."""

import numpy as np
import pytest
import torch

from airtrace.transforms.scaling import RobustScalerTransform, ZScoreTransform
from airtrace.transforms.context import ContextTransform


class TestZScoreTransformEdgeCases:
    """Additional tests for ZScoreTransform to improve coverage."""

    def test_get_stats_not_fitted(self):
        """Test get_stats raises error when not fitted."""
        transform = ZScoreTransform()
        with pytest.raises(RuntimeError, match="not fitted"):
            transform.get_stats()

    def test_set_stats(self):
        """Test set_stats method."""
        # Create and fit a transform
        transform1 = ZScoreTransform()
        
        # Create mock dataset
        class MockDataset:
            def __len__(self):
                return 10
            
            def __getitem__(self, idx):
                x = np.random.randn(32, 5)
                y = np.random.randn(8, 2)
                return {"x": x, "y": y}
        
        dataset = MockDataset()
        transform1.fit(dataset)
        
        # Get stats
        stats = transform1.get_stats()
        
        # Create new transform and set stats
        transform2 = ZScoreTransform()
        transform2.set_stats(stats)
        
        assert transform2.is_fitted
        assert np.allclose(transform2.scaler_x.mean_, transform1.scaler_x.mean_)
        assert np.allclose(transform2.scaler_x.scale_, transform1.scaler_x.scale_)

    def test_call_not_fitted(self):
        """Test __call__ raises error when not fitted."""
        transform = ZScoreTransform()
        x = np.random.randn(32, 5)
        y = np.random.randn(8, 2)
        meta = {}
        
        with pytest.raises(RuntimeError, match="not fitted"):
            transform(x, y, meta)

    def test_zscore_no_center_no_scale(self):
        """Test ZScoreTransform with centering and scaling disabled."""
        transform = ZScoreTransform(center=False, scale=False)
        
        class MockDataset:
            def __len__(self):
                return 10
            
            def __getitem__(self, idx):
                x = np.random.randn(32, 5)
                y = np.random.randn(8, 2)
                return {"x": x, "y": y}
        
        dataset = MockDataset()
        transform.fit(dataset)
        
        x = np.random.randn(32, 5)
        y = np.random.randn(8, 2)
        meta = {}
        
        x_t, y_t, meta_t = transform(x, y, meta)
        
        # With no centering or scaling, output should be same as input
        assert np.allclose(x_t, x)
        assert np.allclose(y_t, y)


class TestRobustScalerTransformEdgeCases:
    """Additional tests for RobustScalerTransform to improve coverage."""

    def test_call_not_fitted(self):
        """Test __call__ raises error when not fitted."""
        transform = RobustScalerTransform()
        x = np.random.randn(32, 5)
        y = np.random.randn(8, 2)
        meta = {}
        
        with pytest.raises(RuntimeError, match="not fitted"):
            transform(x, y, meta)

    def test_custom_quantile_range(self):
        """Test RobustScalerTransform with custom quantile range."""
        transform = RobustScalerTransform(quantile_range=(10.0, 90.0))
        
        class MockDataset:
            def __len__(self):
                return 20
            
            def __getitem__(self, idx):
                x = np.random.randn(32, 5)
                y = np.random.randn(8, 2)
                return {"x": x, "y": y}
        
        dataset = MockDataset()
        transform.fit(dataset)
        
        x = np.random.randn(32, 5)
        y = np.random.randn(8, 2)
        meta = {}
        
        x_t, y_t, meta_t = transform(x, y, meta)
        
        # Should transform without error
        assert x_t.shape == x.shape
        assert y_t.shape == y.shape


class TestContextTransformEdgeCases:
    """Additional tests for ContextTransform to improve coverage."""

    def test_get_stats_not_fitted(self):
        """Test get_stats raises error when not fitted."""
        transform = ContextTransform(use_static=["aircraft_type"])
        with pytest.raises(RuntimeError, match="not fitted"):
            transform.get_stats()

    def test_set_stats(self):
        """Test set_stats method."""
        transform = ContextTransform(use_static=["aircraft_type"])
        
        stats = {
            'static_encoders': {'aircraft_type': {'A320': 0, 'B737': 1}},
            'use_static': ['aircraft_type'],
            'use_plan_deltas': False,
            'use_env': False,
        }
        
        transform.set_stats(stats)
        
        assert transform.is_fitted
        assert transform.static_encoders == stats['static_encoders']

    def test_context_with_plan_deltas(self):
        """Test context transform with plan deltas."""
        transform = ContextTransform(use_plan_deltas=True)
        transform.is_fitted = True
        
        x = np.random.randn(32, 5)
        y = np.random.randn(8, 2)
        plan_deltas = np.random.randn(32, 3)
        meta = {"plan_deltas": plan_deltas}
        
        x_t, y_t, meta_t = transform(x, y, meta)
        
        # Should have added plan deltas
        assert x_t.shape[1] == x.shape[1] + plan_deltas.shape[1]
        assert meta_t["context_dim"] == plan_deltas.shape[1]

    def test_context_with_env_vars(self):
        """Test context transform with environmental variables."""
        transform = ContextTransform(use_env=True)
        transform.is_fitted = True
        
        x = np.random.randn(32, 5)
        y = np.random.randn(8, 2)
        env_vars = np.random.randn(32, 2)
        meta = {"env_vars": env_vars}
        
        x_t, y_t, meta_t = transform(x, y, meta)
        
        # Should have added env vars
        assert x_t.shape[1] == x.shape[1] + env_vars.shape[1]
        assert meta_t["context_dim"] == env_vars.shape[1]

    def test_context_missing_metadata(self):
        """Test context transform with missing metadata."""
        transform = ContextTransform(use_static=["aircraft_type"])
        transform.is_fitted = True
        transform.static_encoders = {"aircraft_type": {"A320": 0, "B737": 1}}
        
        x = np.random.randn(32, 5)
        y = np.random.randn(8, 2)
        meta = {}  # Missing aircraft_type
        
        x_t, y_t, meta_t = transform(x, y, meta)
        
        # Should use default value (0.0)
        assert x_t.shape[1] == x.shape[1] + 1
        assert meta_t["context_dim"] == 1

    def test_context_numeric_feature(self):
        """Test context transform with numeric (non-categorical) feature."""
        transform = ContextTransform(use_static=["altitude"])
        transform.is_fitted = True
        transform.static_encoders = {}  # No encoder for altitude (numeric)
        
        x = np.random.randn(32, 5)
        y = np.random.randn(8, 2)
        meta = {"altitude": 35000.0}
        
        x_t, y_t, meta_t = transform(x, y, meta)
        
        # Should add numeric value as-is
        assert x_t.shape[1] == x.shape[1] + 1
        assert x_t[0, -1] == 35000.0

    def test_context_dim_property(self):
        """Test context_dim property."""
        transform = ContextTransform(use_static=["a", "b", "c"])
        assert transform.context_dim == 3
        
        transform2 = ContextTransform(use_static=[])
        assert transform2.context_dim == 0
