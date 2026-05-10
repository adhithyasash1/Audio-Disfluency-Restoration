from disfluency.position import position_prior


def test_endpoints():
    assert position_prior(0, 10) == 1.0
    assert position_prior(9, 10) == 0.0


def test_monotone_decreasing():
    vals = [position_prior(i, 10) for i in range(10)]
    assert vals == sorted(vals, reverse=True)


def test_single_token():
    assert position_prior(0, 1) == 1.0
    assert position_prior(0, 0) == 1.0
