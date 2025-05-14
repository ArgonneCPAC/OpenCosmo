from opencosmo.spatial.region import BoxRegion


def test_contains():
    reg1 = BoxRegion((100, 100, 100), 15)
    reg2 = BoxRegion((90, 90, 90), 2.5)
    reg3 = BoxRegion((90, 90, 90), 10)
    assert reg1.contains(reg2)
    assert not reg1.contains(reg3)
    assert not reg2.contains(reg1)


def test_interesects():
    reg1 = BoxRegion((100, 100, 100), 20)
    reg2 = BoxRegion((90, 90, 90), 20)
    assert reg1.intersects(reg2)
    assert reg2.intersects(reg1)


def test_neither():
    reg1 = BoxRegion((100, 100, 100), 15)
    reg2 = BoxRegion((75, 75, 75), 5)
    assert not reg1.intersects(reg2)
    assert not reg2.intersects(reg1)
    assert not reg1.contains(reg2)
    assert not reg2.contains(reg1)
