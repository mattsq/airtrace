"""Tests for notebook utilities."""

from unittest.mock import MagicMock, patch

import matplotlib
import pytest

# Use non-interactive backend for testing
matplotlib.use('Agg')

from airtrace.viz.notebooks_utils import setup_notebook


class TestSetupNotebook:
    """Tests for setup_notebook function."""

    def test_setup_notebook_basic(self, capsys):
        """Test basic notebook setup."""
        setup_notebook()
        
        captured = capsys.readouterr()
        assert "Notebook environment set up successfully!" in captured.out

    def test_setup_notebook_sets_rcparams(self):
        """Test that setup_notebook sets matplotlib rcParams."""
        import matplotlib.pyplot as plt
        
        setup_notebook()
        
        # Check that figure size was set (rcParams returns a list, not tuple)
        figsize = plt.rcParams['figure.figsize']
        assert figsize[0] == 12
        assert figsize[1] == 6
        assert plt.rcParams['font.size'] == 10

    def test_setup_notebook_with_ipython(self, capsys):
        """Test setup_notebook with IPython available."""
        mock_ipython = MagicMock()
        
        # Patch get_ipython in the IPython module, not in our module
        with patch('IPython.get_ipython', return_value=mock_ipython):
            setup_notebook()
            
            # Should call magic method
            mock_ipython.magic.assert_called_once_with('matplotlib inline')
        
        captured = capsys.readouterr()
        assert "successfully" in captured.out

    def test_setup_notebook_without_ipython(self, capsys):
        """Test setup_notebook without IPython (non-notebook environment)."""
        with patch('IPython.get_ipython', return_value=None):
            setup_notebook()
        
        # Should still succeed
        captured = capsys.readouterr()
        assert "successfully" in captured.out

    def test_setup_notebook_import_error(self, capsys):
        """Test setup_notebook when IPython import fails."""
        # Mock the import to fail
        import sys
        with patch.dict('sys.modules', {'IPython': None}):
            setup_notebook()
        
        # Should handle gracefully
        captured = capsys.readouterr()
        assert "successfully" in captured.out
