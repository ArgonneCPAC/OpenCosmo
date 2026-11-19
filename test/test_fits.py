from __future__ import annotations

import subprocess
import sys
import textwrap

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from opencosmo.io import write_cutout


@pytest.fixture
def image_hdulist():
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [123.4, -20.0]
    wcs.wcs.crpix = [32.5, 32.5]
    wcs.wcs.cdelt = [-1 / 64, 1 / 64]
    header = wcs.to_header()
    data = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)
    return fits.HDUList(
        [
            fits.PrimaryHDU(header=header),
            fits.ImageHDU(data=data, header=header, name="MAP"),
        ]
    )


@pytest.fixture
def row():
    return {
        "z": 0.5,
        "ra": 123.4 * u.deg,
        "dec": -20.0 * u.deg,
        "name": "halo-1",
        "flag": True,
        "vector": np.array([1.0, 2.0, 3.0]),
    }


def test_write_map_only_hdulist(image_hdulist, tmp_path):
    path = tmp_path / "map.fits"
    write_cutout(path, image_hdulist)

    with fits.open(path) as hdul:
        assert len(hdul) == 2
        assert hdul[1].name == "MAP"
        assert hdul[1].header["CTYPE1"] == "RA---TAN"
        assert np.array_equal(hdul[1].data, image_hdulist[1].data)
    assert len(image_hdulist) == 2


def test_write_cutout_row_and_center_table(image_hdulist, row, tmp_path):
    path = tmp_path / "cutout.fits"
    write_cutout(path, (row, image_hdulist))

    with fits.open(path) as hdul:
        assert len(hdul) == 3
        assert isinstance(hdul["CENTER"], fits.BinTableHDU)

    table = Table.read(path, hdu="CENTER")
    assert len(table) == 1
    # metadata columns are stored sorted by name
    assert list(table.columns) == sorted(row)
    assert table["ra"][0] == pytest.approx(123.4)
    assert table["ra"].unit == u.deg
    assert table["dec"][0] == pytest.approx(-20.0)
    assert table["dec"].unit == u.deg
    assert table["z"][0] == pytest.approx(0.5)
    assert table["name"][0] == "halo-1"
    assert table["flag"][0] == np.True_
    assert np.array_equal(table["vector"][0], [1.0, 2.0, 3.0])


def test_write_does_not_mutate_input_hdulist(image_hdulist, row, tmp_path):
    image_hdulist[0].header["EXTEND"] = False
    original_hdus = list(image_hdulist)
    original_headers = [hdu.header.copy() for hdu in image_hdulist]
    original_data = image_hdulist[1].data.copy()
    write_cutout(tmp_path / "cutout.fits", (row, image_hdulist))

    assert len(image_hdulist) == 2
    assert list(image_hdulist) == original_hdus
    assert all(
        hdu.header == header
        for hdu, header in zip(image_hdulist, original_headers, strict=True)
    )
    np.testing.assert_array_equal(image_hdulist[1].data, original_data)
    # caller HDUs are left open and usable
    assert image_hdulist[1].data is not None


def test_write_refuses_existing_path(image_hdulist, tmp_path):
    path = tmp_path / "cutout.fits"
    write_cutout(path, image_hdulist)

    with pytest.raises(FileExistsError):
        write_cutout(path, image_hdulist)

    write_cutout(path, image_hdulist, overwrite=True)
    with fits.open(path) as hdul:
        assert len(hdul) == 2


def test_write_overwrite_replaces_file(image_hdulist, row, tmp_path):
    path = tmp_path / "cutout.fits"
    write_cutout(path, image_hdulist)
    write_cutout(path, (row, image_hdulist), overwrite=True)

    with fits.open(path) as hdul:
        assert "CENTER" in hdul


def test_write_rejects_empty_row(image_hdulist, tmp_path):
    path = tmp_path / "cutout.fits"
    with pytest.raises(ValueError, match="empty metadata row"):
        write_cutout(path, ({}, image_hdulist))
    assert not path.exists()


def test_write_rejects_invalid_names(image_hdulist, tmp_path):
    with pytest.raises(TypeError, match="must be strings"):
        write_cutout(tmp_path / "a.fits", ({1: 2.0}, image_hdulist))
    with pytest.raises(ValueError, match="non-empty"):
        write_cutout(tmp_path / "b.fits", ({"": 2.0}, image_hdulist))
    with pytest.raises(ValueError, match="68-character"):
        write_cutout(tmp_path / "c.fits", ({"x" * 100: 2.0}, image_hdulist))
    assert not (tmp_path / "a.fits").exists()
    assert not (tmp_path / "b.fits").exists()
    assert not (tmp_path / "c.fits").exists()


def test_write_rejects_unsupported_values(image_hdulist, tmp_path):
    with pytest.raises(TypeError, match="Unsupported type"):
        write_cutout(tmp_path / "a.fits", ({"bad": None}, image_hdulist))
    with pytest.raises(TypeError, match="Unsupported type"):
        write_cutout(tmp_path / "b.fits", ({"bad": {"x": 1}}, image_hdulist))
    with pytest.raises(TypeError, match="object or structured"):
        write_cutout(
            tmp_path / "c.fits",
            ({"bad": np.array([object()], dtype=object)}, image_hdulist),
        )
    with pytest.raises(TypeError, match="object or structured"):
        write_cutout(
            tmp_path / "d.fits",
            ({"bad": np.array([(1.0,)], dtype=[("x", np.float64)])}, image_hdulist),
        )
    assert not (tmp_path / "a.fits").exists()
    assert not (tmp_path / "b.fits").exists()
    assert not (tmp_path / "c.fits").exists()
    assert not (tmp_path / "d.fits").exists()


def test_write_rejects_bad_to_write(image_hdulist, row, tmp_path):
    with pytest.raises(TypeError, match="HDUList"):
        write_cutout(tmp_path / "a.fits", "not an hdulist")
    with pytest.raises(TypeError, match="mapping"):
        write_cutout(tmp_path / "b.fits", (row["z"], image_hdulist))
    with pytest.raises(TypeError, match="HDUList"):
        write_cutout(tmp_path / "c.fits", (row, "not an hdulist"))
    with pytest.raises(ValueError, match="exactly"):
        write_cutout(tmp_path / "d.fits", (row, image_hdulist, None))
    for name in "abcd":
        assert not (tmp_path / f"{name}.fits").exists()


def test_write_rejects_center_collision(row, tmp_path):
    hdulist = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.zeros((4, 4)), name="CENTER"),
        ]
    )
    path = tmp_path / "cutout.fits"
    with pytest.raises(ValueError, match="CENTER"):
        write_cutout(path, (row, hdulist))
    assert not path.exists()


def test_fits_module_has_no_top_level_astropy_fits_import():
    """Statically, the writer module must not import astropy.io.fits or
    astropy.table at module level."""
    import ast
    from pathlib import Path

    import opencosmo.io.fits as fits_module

    tree = ast.parse(Path(fits_module.__file__).read_text())
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            assert not node.module.startswith("astropy"), (
                f"top-level import of {node.module} in opencosmo/io/fits.py"
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("astropy"), (
                    f"top-level import of {alias.name} in opencosmo/io/fits.py"
                )


def test_fits_import_is_lazy(tmp_path):
    """Loading the writer module directly (without the rest of the opencosmo
    package, whose other dependencies may import astropy.io.fits) must not
    import astropy.io.fits, and the filesystem check must happen before the
    lazy import."""
    import opencosmo.io.fits as fits_module

    existing = tmp_path / "exists.fits"
    existing.write_bytes(b"")
    script = textwrap.dedent(
        """
        import importlib.abc
        import importlib.util
        import sys


        class FitsBlocker(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "astropy.io.fits" or name.startswith("astropy.io.fits."):
                    raise ImportError(f"blocked import: {name}")
                return None


        sys.meta_path.insert(0, FitsBlocker())
        for mod in list(sys.modules):
            if mod == "astropy.io.fits" or mod.startswith("astropy.io.fits."):
                del sys.modules[mod]

        spec = importlib.util.spec_from_file_location("oc_fits_writer", sys.argv[2])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert "astropy.io.fits" not in sys.modules, (
            "astropy.io.fits was imported eagerly"
        )

        try:
            module.write_cutout(sys.argv[1], "not an hdulist")
        except FileExistsError:
            pass
        else:
            raise AssertionError("expected FileExistsError before lazy import")

        assert "astropy.io.fits" not in sys.modules, (
            "astropy.io.fits was imported by the filesystem check"
        )
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(existing), fits_module.__file__],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
