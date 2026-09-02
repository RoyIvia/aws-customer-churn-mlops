"""Tests for model evaluation and quality gating."""

import pandas as pd
import pytest

from src.evaluation.evaluate import (
    apply_quality_gate,
    calculate_metrics,
)


def test_calculate_metrics():
    y_true = pd.Series([0, 0, 1, 1])
    probabilities = [0.1, 0.2, 0.8, 0.9]

    metrics = calculate_metrics(
        y_true,
        probabilities,
        threshold=0.5,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_quality_gate_passes():
    metrics = {
        "f1": 0.70,
        "recall": 0.80,
        "roc_auc": 0.85,
    }

    result = apply_quality_gate(metrics)

    assert result["status"] == "PASS"
    assert result["passed"] is True


def test_quality_gate_fails():
    metrics = {
        "f1": 0.50,
        "recall": 0.60,
        "roc_auc": 0.70,
    }

    result = apply_quality_gate(metrics)

    assert result["status"] == "FAIL"
    assert result["passed"] is False


def test_quality_gate_requires_all_metrics():
    metrics = {
        "f1": 0.70,
        "recall": 0.80,
        "roc_auc": 0.70,
    }

    result = apply_quality_gate(metrics)

    assert result["status"] == "FAIL"
    assert result["checks"]["f1"] is True
    assert result["checks"]["recall"] is True
    assert result["checks"]["roc_auc"] is False


def test_metrics_include_confusion_matrix():
    y_true = pd.Series([0, 0, 1, 1])
    probabilities = [0.1, 0.8, 0.7, 0.2]

    metrics = calculate_metrics(
        y_true,
        probabilities,
        threshold=0.5,
    )

    confusion = metrics["confusion_matrix"]

    assert set(confusion.keys()) == {
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    }
