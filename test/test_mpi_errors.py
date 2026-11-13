from __future__ import annotations

import shutil
import subprocess
import sys
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, call

import opencosmo.io.mpi as io_mpi
import pytest
from opencosmo.io.schema import FileEntry, make_schema
from opencosmo.mpi import abort_on_error, get_mpi

if TYPE_CHECKING:
    from pathlib import Path

    from opencosmo.io.schema import Schema

SUBPROCESS_TIMEOUT = 60

_DISTINCTIVE_MESSAGE = "distinctive-abort-on-error-rank0-failure"

_ABORT_CHILD = f"""
from mpi4py import MPI

from opencosmo.mpi import abort_on_error

comm = MPI.COMM_WORLD
with abort_on_error(comm):
    if comm.Get_rank() == 0:
        raise ValueError("{_DISTINCTIVE_MESSAGE}")
    comm.Barrier()
"""

_SUCCESS_CHILD = """
from mpi4py import MPI

from opencosmo.mpi import abort_on_error

comm = MPI.COMM_WORLD
with abort_on_error(comm):
    comm.Barrier()
"""


def test_abort_on_error_none_success() -> None:
    with abort_on_error(None):
        pass


def test_abort_on_error_none_preserves_original_exception() -> None:
    error = ValueError("serial boom")
    with pytest.raises(ValueError) as excinfo, abort_on_error(None):
        raise error
    assert excinfo.value is error


@pytest.fixture
def mpi_launcher() -> str:
    MPI = get_mpi()
    if MPI is None:
        pytest.skip("mpi4py is not available")
    if MPI.COMM_WORLD.Get_size() > 1:
        pytest.skip("pytest is already running under multi-rank MPI")
    mpiexec = shutil.which("mpiexec")
    if mpiexec is None:
        pytest.skip("mpiexec launcher is not available")
    return mpiexec


def _run_child(
    mpiexec: str, source: str, tmp_path: Path
) -> subprocess.CompletedProcess[str]:
    script = tmp_path / "child.py"
    script.write_text(source)
    return subprocess.run(
        [mpiexec, "-n", "2", sys.executable, str(script)],
        check=False,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT,
    )


def test_abort_on_error_mpi_aborts_on_rank_error(
    mpi_launcher: str, tmp_path: Path
) -> None:
    result = _run_child(mpi_launcher, _ABORT_CHILD, tmp_path)
    assert result.returncode != 0
    assert "rank 0" in result.stderr
    assert "ValueError" in result.stderr
    assert _DISTINCTIVE_MESSAGE in result.stderr
    assert "Traceback" in result.stderr


def test_abort_on_error_mpi_success_exits_zero(
    mpi_launcher: str, tmp_path: Path
) -> None:
    result = _run_child(mpi_launcher, _SUCCESS_CHILD, tmp_path)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def write_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Mock, Mock, Schema, list[str]]:
    """Isolate write dispatch and context ordering from schema preparation."""
    events: list[str] = []
    comm = Mock()
    comm.Get_rank.return_value = 0
    comm.allgather.side_effect = lambda value: [value]
    comm.Create.return_value.Get_rank.return_value = 0
    monkeypatch.setattr(io_mpi, "get_comm_world", Mock(return_value=comm))
    monkeypatch.setattr(io_mpi, "get_mpi", Mock(return_value=Mock(COMM_NULL=None)))

    schema = make_schema("lightcone", FileEntry.LIGHTCONE)
    monkeypatch.setattr(io_mpi, "schema_data_length", Mock(return_value=1))
    monkeypatch.setattr(io_mpi, "verify_structure", Mock())
    monkeypatch.setattr(io_mpi, "sync_schemas", Mock(return_value=schema))
    monkeypatch.setattr(io_mpi, "__verify_structure_collective", Mock())
    monkeypatch.setattr(io_mpi, "__get_all_offsets", Mock(return_value={}))
    monkeypatch.setattr(io_mpi, "__allocate", Mock())
    monkeypatch.setattr(io_mpi, "cleanup_mpi", Mock())

    file_context = MagicMock()
    file_context.__enter__.side_effect = lambda: events.append("open") or file_context
    file_context.__exit__.side_effect = lambda *args: events.append("close")
    hdf5 = Mock()
    hdf5.File.return_value = file_context
    hdf5.get_config.return_value.mpi = True
    monkeypatch.setattr(io_mpi, "h5py", hdf5)
    return comm, hdf5, schema, events


def test_write_parallel_null_comm_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(io_mpi, "get_comm_world", Mock(return_value=None))
    schema = make_schema("lightcone", FileEntry.LIGHTCONE)
    with pytest.raises(ValueError, match="null comm"):
        io_mpi.write_parallel(tmp_path / "out.hdf5", schema, False)


def test_write_parallel_aborts_preparation_errors_on_full_comm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    write_environment: tuple[Mock, Mock, Schema, list[str]],
) -> None:
    comm, _, schema, _ = write_environment
    comm.Abort.side_effect = RuntimeError("simulated MPI abort")
    prepare = Mock(side_effect=ValueError("preparation failure"))
    monkeypatch.setattr(io_mpi, "__do_parallel_write", prepare)
    path = tmp_path / "out.hdf5"

    with pytest.raises(RuntimeError, match="simulated MPI abort"):
        io_mpi.write_parallel(path, schema, False)

    prepare.assert_called_once_with(path, schema, False, comm, [0])
    comm.Abort.assert_called_once_with(1)
    comm.Create.return_value.Abort.assert_not_called()


def test_empty_write_raises_without_aborting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    write_environment: tuple[Mock, Mock, Schema, list[str]],
) -> None:
    comm, hdf5, schema, _ = write_environment
    monkeypatch.setattr(io_mpi, "schema_data_length", Mock(return_value=0))

    with pytest.raises(ValueError, match="No ranks have any data"):
        io_mpi.write_parallel(tmp_path / "empty.hdf5", schema, False)

    comm.Abort.assert_not_called()
    comm.Create.assert_not_called()
    hdf5.File.assert_not_called()


@pytest.mark.parametrize("local_invalid", [True, False])
def test_invalid_schema_raises_without_aborting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    write_environment: tuple[Mock, Mock, Schema, list[str]],
    local_invalid: bool,
) -> None:
    comm, hdf5, schema, _ = write_environment
    error = ValueError("invalid local schema")
    if local_invalid:
        monkeypatch.setattr(io_mpi, "verify_structure", Mock(side_effect=error))
    else:
        comm.allgather.side_effect = [
            [tmp_path / "out.hdf5"] * 2,
            [io_mpi.CombineState.VALID, io_mpi.CombineState.INVALID],
        ]
    message = "invalid local schema" if local_invalid else "invalid schemas"

    with pytest.raises(ValueError, match=message) as excinfo:
        io_mpi.write_parallel(tmp_path / "out.hdf5", schema, False)

    if local_invalid:
        assert excinfo.value is error
    comm.Abort.assert_not_called()
    comm.Create.assert_not_called()
    hdf5.File.assert_not_called()


def test_inconsistent_paths_raise_without_aborting(
    tmp_path: Path,
    write_environment: tuple[Mock, Mock, Schema, list[str]],
) -> None:
    comm, hdf5, schema, _ = write_environment
    comm.allgather.return_value = [tmp_path / "a.hdf5", tmp_path / "b.hdf5"]
    comm.allgather.side_effect = None

    with pytest.raises(ValueError, match="Different ranks recieved a different path"):
        io_mpi.write_parallel(tmp_path / "a.hdf5", schema, False)

    comm.Abort.assert_not_called()
    comm.Create.assert_not_called()
    hdf5.File.assert_not_called()


def test_unexpected_validation_error_still_aborts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    write_environment: tuple[Mock, Mock, Schema, list[str]],
) -> None:
    comm, hdf5, schema, _ = write_environment
    comm.Abort.side_effect = RuntimeError("simulated MPI abort")
    monkeypatch.setattr(
        io_mpi, "verify_structure", Mock(side_effect=RuntimeError("unexpected failure"))
    )

    with pytest.raises(RuntimeError, match="simulated MPI abort"):
        io_mpi.write_parallel(tmp_path / "out.hdf5", schema, False)

    comm.Abort.assert_called_once_with(1)
    comm.Create.assert_not_called()
    hdf5.File.assert_not_called()


def test_parallel_hdf5_write_uses_data_comm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    write_environment: tuple[Mock, Mock, Schema, list[str]],
) -> None:
    comm, hdf5, schema, events = write_environment
    write = Mock(side_effect=lambda *args: events.append("write"))
    monkeypatch.setattr(io_mpi, "__write_parallel", write)
    path = tmp_path / "out.hdf5"

    io_mpi.write_parallel(path, schema, False)

    data_comm = comm.Create.return_value
    assert hdf5.File.call_args_list == [
        call(path, "w"),
        call(path, "a", driver="mpio", comm=data_comm),
    ]
    write.assert_called_once_with(schema, hdf5.File.return_value, {}, data_comm)
    assert events == ["open", "close", "open", "write", "close"]
    comm.Abort.assert_not_called()


def test_write_error_aborts_before_file_close_without_serial_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    write_environment: tuple[Mock, Mock, Schema, list[str]],
) -> None:
    comm, _, schema, events = write_environment

    def simulate_abort(code: int) -> None:
        events.append("abort")
        raise RuntimeError("simulated MPI abort")

    comm.Abort.side_effect = simulate_abort
    monkeypatch.setattr(
        io_mpi, "__write_parallel", Mock(side_effect=ValueError("write failure"))
    )
    replace_writers = Mock()
    write_serial = Mock()
    monkeypatch.setattr(io_mpi, "__replace_writers_with_updates", replace_writers)
    monkeypatch.setattr(io_mpi, "__write_serial", write_serial)

    # Exercise the inner guard independently: a real Abort never returns,
    # whereas the simulated abort raises and would reach the outer guard too.
    with pytest.raises(RuntimeError, match="simulated MPI abort"):
        io_mpi.__do_parallel_write(tmp_path / "out.hdf5", schema, False, comm, [0])

    assert events == ["open", "close", "open", "abort", "close"]
    comm.Abort.assert_called_once_with(1)
    comm.Create.return_value.Abort.assert_not_called()
    replace_writers.assert_not_called()
    write_serial.assert_not_called()
    io_mpi.cleanup_mpi.assert_not_called()


def test_non_mpi_hdf5_selects_serial_fallback_without_mpio_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    write_environment: tuple[Mock, Mock, Schema, list[str]],
) -> None:
    comm, hdf5, schema, events = write_environment
    hdf5.get_config.return_value.mpi = False
    replace_writers = Mock(return_value=schema)
    write_serial = Mock()
    write_parallel = Mock()
    monkeypatch.setattr(io_mpi, "__replace_writers_with_updates", replace_writers)
    monkeypatch.setattr(io_mpi, "__write_serial", write_serial)
    monkeypatch.setattr(io_mpi, "__write_parallel", write_parallel)
    path = tmp_path / "out.hdf5"

    io_mpi.write_parallel(path, schema, False)

    data_comm = comm.Create.return_value
    hdf5.File.assert_called_once_with(path, "w")
    replace_writers.assert_called_once_with(schema, data_comm)
    write_serial.assert_called_once_with(schema, path, {}, data_comm)
    write_parallel.assert_not_called()
    comm.Abort.assert_not_called()
    assert events == ["open", "close"]
