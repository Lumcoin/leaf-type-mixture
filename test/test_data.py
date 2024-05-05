from ltm.data import combine_band_name, split_band_name


def test_combine_band_name():
    composite_idx = 1
    band_label = "TCI_G"
    reducer = "kendallsCorrelation"
    reducer_band = "p-value"
    combined = combine_band_name(
        composite_idx, band_label, reducer, reducer_band
    )
    assert combined == "1 TCI_G kendallsCorrelation p-value"


def test_split_band_name():
    band_name = "1 TCI_G kendallsCorrelation p-value"
    a, b, c, d = split_band_name(band_name)
    assert a == 1
    assert b == "TCI_G"
    assert c == "kendallsCorrelation"
    assert d == "p-value"
