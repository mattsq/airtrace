
import unittest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from airtrace.export.transform_wrappers import (
    DifferenceWrapper,
    TransformCompose,
    create_transform_wrapper,
    create_forward_transform_pipeline,
    create_inverse_transform_pipeline,
    _extract_transform_index
)

class TestTransformWrappersCoverage(unittest.TestCase):
    
    def test_difference_wrapper_forward_batched(self):
        # [B=2, T=3, D=1]
        x = torch.tensor([[[1.0], [2.0], [4.0]], [[10.0], [12.0], [15.0]]])
        wrapper = DifferenceWrapper(order=1)
        out = wrapper(x)
        
        # Expected:
        # B1: [1.0, 2-1=1, 4-2=2]
        # B2: [10.0, 12-10=2, 15-12=3]
        expected = torch.tensor([[[1.0], [1.0], [2.0]], [[10.0], [2.0], [3.0]]])
        torch.testing.assert_close(out, expected)

    def test_difference_wrapper_forward_unbatched(self):
        # [T=3, D=1]
        x = torch.tensor([[1.0], [2.0], [4.0]])
        wrapper = DifferenceWrapper(order=1)
        out = wrapper(x)
        
        # Expected: [1.0, 1.0, 2.0]
        expected = torch.tensor([[1.0], [1.0], [2.0]])
        torch.testing.assert_close(out, expected)

    def test_difference_wrapper_inverse_batched(self):
        # [B=2, T=3, D=1]
        x_diff = torch.tensor([[[1.0], [1.0], [2.0]], [[10.0], [2.0], [3.0]]])
        wrapper = DifferenceWrapper(order=1)
        out = wrapper.inverse(x_diff)
        
        # Expected:
        # B1: [1, 1+1=2, 2+2=4]
        # B2: [10, 10+2=12, 12+3=15]
        expected = torch.tensor([[[1.0], [2.0], [4.0]], [[10.0], [12.0], [15.0]]])
        torch.testing.assert_close(out, expected)

    def test_difference_wrapper_inverse_unbatched(self):
        # [T=3, D=1]
        x_diff = torch.tensor([[1.0], [1.0], [2.0]])
        wrapper = DifferenceWrapper(order=1)
        out = wrapper.inverse(x_diff)
        
        expected = torch.tensor([[1.0], [2.0], [4.0]])
        torch.testing.assert_close(out, expected)

    def test_difference_wrapper_order_2(self):
        # x: [1, 4, 9, 16] (squares)
        # diff1: [1, 3, 5, 7]
        # diff2: [1, 2, 2, 2] (pad preserves first element of previous diff)
        
        # Implementation details:
        # x_diff starts as x.
        # Loop 1:
        #   diff = x[1:] - x[:-1]
        #   x_diff = cat(x[:1], diff) -> [1, 3, 5, 7]
        # Loop 2:
        #   diff = [3-1=2, 5-3=2, 7-5=2]
        #   x_diff = cat([1], [2, 2, 2]) -> [1, 2, 2, 2]
        
        x = torch.tensor([[1.0], [4.0], [9.0], [16.0]])
        wrapper = DifferenceWrapper(order=2)
        out = wrapper(x)
        expected = torch.tensor([[1.0], [2.0], [2.0], [2.0]])
        torch.testing.assert_close(out, expected)
        
        inv = wrapper.inverse(out)
        torch.testing.assert_close(inv, x)

    def test_transform_compose_inverse(self):
        # t1: x + 1
        # t2: x * 2
        # forward: (x+1)*2
        # inverse: (y/2)-1
        
        class AddOne(torch.nn.Module):
            def forward(self, x): return x + 1
            def inverse(self, x): return x - 1
            
        class MulTwo(torch.nn.Module):
            def forward(self, x): return x * 2
            def inverse(self, x): return x / 2
            
        pipeline = TransformCompose([AddOne(), MulTwo()])
        x = torch.tensor([1.0])
        
        # Forward
        y = pipeline(x)
        self.assertEqual(y.item(), 4.0)
        
        # Inverse
        x_rec = pipeline.inverse(y)
        self.assertEqual(x_rec.item(), 1.0)

    def test_create_transform_wrapper_unsupported(self):
        stats = {}
        wrapper = create_transform_wrapper(stats, "UnknownTransform_0")
        self.assertIsNone(wrapper)

    def test_create_transform_wrapper_zscore_missing_stats(self):
        stats = {"scaler_x_mean": np.array([0])} # missing scale
        wrapper = create_transform_wrapper(stats, "ZScoreTransform_0", use_x_scaler=True)
        self.assertIsNone(wrapper)

    def test_create_transform_wrapper_robust_missing_stats(self):
        stats = {"scaler_x_center": np.array([0])} # missing scale
        wrapper = create_transform_wrapper(stats, "RobustScalerTransform_0", use_x_scaler=True)
        self.assertIsNone(wrapper)

    def test_extract_transform_index(self):
        self.assertEqual(_extract_transform_index("T_0"), 0)
        self.assertEqual(_extract_transform_index("T_10"), 10)
        self.assertEqual(_extract_transform_index("NoIndex"), 0) # Fallback

    def test_create_pipeline_ordering(self):
        stats = {
            "T_1": {"order": 1},
            "T_0": {"order": 1}
        }
        
        class MockModule(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                
        with patch("airtrace.export.transform_wrappers.create_transform_wrapper") as mock_create:
            mock_create.side_effect = lambda s, name, use_x_scaler: MockModule()
            
            pipeline = create_forward_transform_pipeline(stats)
            
            # Check call order
            calls = mock_create.call_args_list
            self.assertEqual(calls[0][0][1], "T_0")
            self.assertEqual(calls[1][0][1], "T_1")

    def test_create_pipeline_empty(self):
        pipeline = create_forward_transform_pipeline({})
        self.assertIsInstance(pipeline, torch.nn.Identity)

    def test_create_pipeline_inverse_ordering(self):
        # Inverse pipeline creation also iterates in sorted index order (0 then 1).
        # TransformCompose.inverse then applies them in reverse (1 then 0).
        stats = {
            "T_1": {"order": 1},
            "T_0": {"order": 1}
        }
        
        class MockModule(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

        with patch("airtrace.export.transform_wrappers.create_transform_wrapper") as mock_create:
            mock_create.side_effect = lambda s, name, use_x_scaler: MockModule()
            pipeline = create_inverse_transform_pipeline(stats)
            
            calls = mock_create.call_args_list
            self.assertEqual(calls[0][0][1], "T_0")
            self.assertEqual(calls[1][0][1], "T_1")
