import pytest
import opencosmo as oc
import matplotlib.pyplot as plt

@pytest.fixture
def input_path(data_path):
    return data_path / "haloproperties.hdf5"

def test_spatial_idx(input_path):
    ds = oc.read(input_path, "scalefree")
    ds1 = ds.spatial_query((60, 60, 60), (70, 70, 70))

    ds2 = ds.filter(oc.col("fof_halo_center_x") > 60, oc.col("fof_halo_center_x") < 70, \
        oc.col("fof_halo_center_y") > 60, oc.col("fof_halo_center_y") < 70, \
        oc.col("fof_halo_center_z") > 60, oc.col("fof_halo_center_z") < 70)
    
    tags1 = ds1.data["fof_halo_tag"]
    tags2 = ds2.data["fof_halo_tag"]
    assert set(tags1) == set(tags2)

