import numpy as np
import matplotlib.pyplot as plt


class TestCurlyBrace:
    """Test the curlyBrace function."""

    def test_basic_functionality(self):
        """Test basic curlyBrace plotting."""
        from curlyBrace import curlyBrace

        fig, ax = plt.subplots()

        # Plot a simple line
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)

        # Add a curly brace
        p1 = [0.0, 0.0]
        p2 = [np.pi, 0.0]

        result = curlyBrace(fig, ax, p1, p2, k_r=0.1, bool_auto=True)

        # curlyBrace returns (theta, summit, arc1, arc2, arc3, arc4)
        assert len(result) == 6

        theta, summit, arc1, arc2, arc3, arc4 = result

        # Check that theta is a float
        assert isinstance(theta, (float, np.floating))

        # Check that summit is a list with 2 elements
        assert len(summit) == 2

        # Check that arcs are lists of lists
        assert len(arc1) == 2
        assert len(arc2) == 2
        assert len(arc3) == 2
        assert len(arc4) == 2

        plt.close(fig)

    def test_with_text_annotation(self):
        """Test curlyBrace with text annotation."""
        from curlyBrace import curlyBrace

        fig, ax = plt.subplots()

        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)

        p1 = [0.0, 0.0]
        p2 = [np.pi, 0.0]

        result = curlyBrace(
            fig, ax, p1, p2,
            k_r=0.1,
            bool_auto=True,
            str_text='Test',
            fontdict={'size': 12}
        )

        assert len(result) == 6

        plt.close(fig)

    def test_auto_scale_off(self):
        """Test curlyBrace with auto scale turned off."""
        from curlyBrace import curlyBrace

        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        x = np.linspace(0, 2*np.pi, 100)
        y = np.sin(x)
        ax.plot(x, y)

        p1 = [0.0, 0.0]
        p2 = [np.pi, 0.0]

        result = curlyBrace(fig, ax, p1, p2, k_r=0.1, bool_auto=False)

        assert len(result) == 6

        plt.close(fig)
