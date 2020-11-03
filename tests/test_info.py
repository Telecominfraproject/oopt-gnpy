
import pytest
from numpy import array, zeros, ones
from numpy.testing import assert_array_equal
from gnpy.core.info import dimension_reshape, create_arbitrary_spectral_information
from gnpy.core.exceptions import InfoError


def test_dim_reshape():
    assert_array_equal(dimension_reshape(value=[1, 2, 3], dimension=3), array([1, 2, 3]))
    assert_array_equal(dimension_reshape(value=[1], dimension=3), array([1, 1, 1]))
    assert_array_equal(dimension_reshape(value=1, dimension=1), array([1]))
    assert_array_equal(dimension_reshape(value=1, dimension=3), array([1, 1, 1]))
    assert_array_equal(dimension_reshape(value=None, dimension=2, default=[1, 2]), array([1, 2]))
    assert_array_equal(dimension_reshape(value=None, dimension=3, default=1), array([1, 1, 1]))
    with pytest.raises(InfoError) as e:
        dimension_reshape(value=[1, 2, 3], dimension=1, name='field_name')
    assert str(e.value) == 'Dimension mismatch field: field_name.'
    with pytest.raises(InfoError) as e:
        dimension_reshape(value=None, dimension=3, default=None, name='field_name')
    assert str(e.value) == 'Missing mandatory field: field_name.'


def test_create_arbitrary_spectral_information():
    frequency = [193.25e12, 193.3e12, 193.35e12]
    baud_rate = 32e9
    signal = [1, 1, 1]
    si = create_arbitrary_spectral_information(frequency=frequency, baud_rate=baud_rate, signal=signal)
    assert_array_equal(si.baud_rate, array([32e9, 32e9, 32e9]))
    assert_array_equal(si.grid, array([37.5e9, 37.5e9, 37.5e9]))
    assert_array_equal(si.signal, ones(3))
    assert_array_equal(si.nli, zeros(3))
    assert_array_equal(si.ase, zeros(3))
    assert_array_equal(si.roll_off, zeros(3))
    assert_array_equal(si.chromatic_dispersion, zeros(3))
    assert_array_equal(si.pmd, zeros(3))
    si.channel_number == [1, 2, 3]
    si.number_of_channels == 3

    with pytest.raises(InfoError) as e:
        si += si
    assert str(e.value) == 'Spectra cannot be summed: channels overlapping.'

    frequency = [193.35e12, 193.3e12, 193.25e12]
    signal = [1, 2, 3]
    si = create_arbitrary_spectral_information(frequency=frequency, baud_rate=baud_rate, signal=signal)
    assert_array_equal(si.signal, array([3, 2, 1]))

    frequency = [193.25e12, 193.3e12, 193.35e12]
    baud_rate = 64e9
    grid = 50e9
    with pytest.raises(InfoError) as e:
        si = create_arbitrary_spectral_information(frequency=frequency, baud_rate=baud_rate, grid=grid)
    assert str(e.value) == 'Spectrum baud rate larger than the grid.'

    frequency = [193.26e12, 193.3e12, 193.34e12]
    baud_rate = 40e9
    with pytest.raises(InfoError) as e:
        si = create_arbitrary_spectral_information(frequency=frequency, baud_rate=baud_rate)
    assert str(e.value) == 'Spectrum required grid larger than the frequencies spectral distance.'
