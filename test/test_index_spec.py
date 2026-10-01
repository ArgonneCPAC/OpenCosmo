from __future__ import annotations

from opencosmo.io.index_spec import empty_ref, full, index_spec_for, spatial


class TestIndexSpecFor:
    """Resolution of a node's index spec from its role and distribution mode."""

    def test_source_in_spatial_mode_is_partitioned(self) -> None:
        assert index_spec_for("spatial", False, is_source=True) is spatial

    def test_non_source_is_full(self) -> None:
        assert index_spec_for("spatial", False, is_source=False) is full

    def test_redshift_step_source_is_full(self) -> None:
        assert index_spec_for("redshift_step", False, is_source=True) is full

    def test_empty_ref_reads_nothing(self) -> None:
        assert index_spec_for("spatial", True, is_source=True) is empty_ref

    def test_replicated_overrides_spatial_partitioning(self) -> None:
        """A replicated node is whole on every rank even in spatial mode."""
        assert (
            index_spec_for("spatial", False, is_source=True, is_replicated=True) is full
        )

    def test_replicated_overrides_empty_ref(self) -> None:
        """is_empty_ref is per-rank and would otherwise empty a replicated node.

        This is the ordering that keeps a companion healpix map intact on reference
        ranks under MpiMode.REDSHIFT.
        """
        assert (
            index_spec_for("spatial", True, is_source=True, is_replicated=True) is full
        )
        assert (
            index_spec_for("redshift_step", True, is_source=False, is_replicated=True)
            is full
        )

    def test_not_replicated_by_default(self) -> None:
        """Omitting is_replicated preserves the pre-existing resolution."""
        assert index_spec_for("spatial", True, is_source=True) is empty_ref
