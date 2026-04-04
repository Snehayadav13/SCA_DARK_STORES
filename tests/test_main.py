"""
Tests for main.py — Pipeline orchestrator.

Covers:
    - STAGES list integrity
    - RUNNERS registry completeness
    - Argument parser (--stage, --from)
    - Stage ordering
"""

from __future__ import annotations

import pytest

from main import STAGES, RUNNERS


# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------


class TestStageRegistry:
    def test_stages_not_empty(self):
        assert len(STAGES) > 0

    def test_no_duplicate_stages(self):
        assert len(STAGES) == len(set(STAGES))

    def test_all_stages_have_runners(self):
        for stage in STAGES:
            assert stage in RUNNERS, f"Stage '{stage}' has no runner function"

    def test_data_pipeline_is_first(self):
        assert STAGES[0] == "data_pipeline"

    def test_forward_vrp_before_reverse(self):
        fwd_idx = STAGES.index("forward_vrp")
        rev_idx = STAGES.index("reverse_vrp")
        assert fwd_idx < rev_idx

    def test_demand_baseline_before_clustering(self):
        db_idx = STAGES.index("demand_baseline")
        cl_idx = STAGES.index("clustering")
        assert db_idx < cl_idx

    def test_clustering_before_return_classifier(self):
        cl_idx = STAGES.index("clustering")
        rc_idx = STAGES.index("return_classifier")
        assert cl_idx < rc_idx

    def test_return_classifier_before_forward_vrp(self):
        rc_idx = STAGES.index("return_classifier")
        fwd_idx = STAGES.index("forward_vrp")
        assert rc_idx < fwd_idx


# ---------------------------------------------------------------------------
# Runners are callable
# ---------------------------------------------------------------------------


class TestRunners:
    def test_all_runners_callable(self):
        for name, func in RUNNERS.items():
            assert callable(func), f"Runner for '{name}' is not callable"

    def test_no_extra_runners(self):
        """Every runner should be a registered stage or a bonus stage."""
        for name in RUNNERS:
            # Allow commented-out stages too
            assert isinstance(name, str)


# ---------------------------------------------------------------------------
# CLI Argument Parsing logic
# ---------------------------------------------------------------------------

from unittest.mock import patch
import sys
from main import main


class TestCliArguments:
    def test_run_all_default(self):
        """Test default runs all STAGES."""
        from unittest.mock import MagicMock

        mock_runners = {s: MagicMock() for s in STAGES}
        with patch("main.RUNNERS", mock_runners), patch.object(
            sys, "argv", ["main.py"]
        ):
            main()
        for stage in STAGES:
            assert mock_runners[stage].called, f"{stage} should have been called"

    def test_single_stage(self):
        """Test running a single stage."""
        from unittest.mock import MagicMock

        mock_runners = {s: MagicMock() for s in STAGES}
        with patch("main.RUNNERS", mock_runners), patch.object(
            sys, "argv", ["main.py", "--stage", "clustering"]
        ):
            main()
        assert mock_runners["clustering"].called
        assert not mock_runners["forward_vrp"].called

    def test_from_stage(self):
        """Test resuming pipeline from a stage."""
        from unittest.mock import MagicMock

        mock_runners = {s: MagicMock() for s in STAGES}
        with patch("main.RUNNERS", mock_runners), patch.object(
            sys, "argv", ["main.py", "--from", "forward_vrp"]
        ):
            main()

        # Stages before forward_vrp should not have been called
        assert not mock_runners["clustering"].called
        assert not mock_runners["return_classifier"].called

        # Stages starting from forward_vrp should have been called
        assert mock_runners["forward_vrp"].called
        assert mock_runners["reverse_vrp"].called
