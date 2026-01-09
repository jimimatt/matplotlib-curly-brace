"""Tests for utility functions."""

import matplotlib.pyplot as plt

from curlyBrace import getAxSize


class TestGetAxSize:
    """Test the getAxSize utility function."""

    def test_import(self) -> None:
        """Test that getAxSize can be imported."""
        from curlyBrace import getAxSize

        assert callable(getAxSize)

    def test_basic_functionality(self) -> None:
        """Test basic getAxSize functionality."""
        fig, ax = plt.subplots(figsize=(8, 6))

        ax_width, ax_height = getAxSize(fig, ax)

        # Check that sizes are positive numbers
        assert isinstance(ax_width, (int, float))
        assert isinstance(ax_height, (int, float))
        assert ax_width > 0
        assert ax_height > 0

        plt.close(fig)

    def test_different_figure_sizes(self) -> None:
        """Test getAxSize with different figure sizes."""
        # Small figure
        fig_small, ax_small = plt.subplots(figsize=(4, 3))
        w_small, h_small = getAxSize(fig_small, ax_small)

        # Large figure
        fig_large, ax_large = plt.subplots(figsize=(12, 9))
        w_large, h_large = getAxSize(fig_large, ax_large)

        # Larger figure should have larger axes
        assert w_large > w_small
        assert h_large > h_small

        plt.close(fig_small)
        plt.close(fig_large)

    def test_aspect_ratio(self) -> None:
        """Test getAxSize maintains expected aspect ratios."""
        # Wide figure
        fig_wide, ax_wide = plt.subplots(figsize=(10, 5))
        w_wide, h_wide = getAxSize(fig_wide, ax_wide)

        # Tall figure
        fig_tall, ax_tall = plt.subplots(figsize=(5, 10))
        w_tall, h_tall = getAxSize(fig_tall, ax_tall)

        # Wide figure should be wider than tall
        assert w_wide > h_wide

        # Tall figure should be taller than wide
        assert h_tall > w_tall

        plt.close(fig_wide)
        plt.close(fig_tall)
