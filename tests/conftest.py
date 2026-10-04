"""Shared pytest fixtures and configuration."""

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")
