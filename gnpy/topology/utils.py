
# @lynx-corenet: functions shifted to this new file to prevent Import Errors because of a circular dependency

def frequency_to_n(freq, grid=0.00625e12):
    """ converts frequency into the n value (ITU grid)
        reference to Recommendation G.694.1 (02/12), Figure I.3
        https://www.itu.int/rec/T-REC-G.694.1-201202-I/en

    >>> frequency_to_n(193.1375e12)
    6
    >>> frequency_to_n(193.225e12)
    20

    """
    return (int)((freq - 193.1e12) / grid)


def nvalue_to_frequency(nvalue, grid=0.00625e12):
    """ converts n value into a frequency
        reference to Recommendation G.694.1 (02/12), Table 1
        https://www.itu.int/rec/T-REC-G.694.1-201202-I/en

    >>> nvalue_to_frequency(6)
    193137500000000.0
    >>> nvalue_to_frequency(-1, 0.1e12)
    193000000000000.0

    """
    return 193.1e12 + nvalue * grid


def mvalue_to_slots(nvalue, mvalue):
    """ convert center n an m into start and stop n
    """
    startn = nvalue - mvalue
    stopn = nvalue + mvalue - 1
    return startn, stopn


def slots_to_m(startn, stopn):
    """ converts the start and stop n values to the center n and m value
        reference to Recommendation G.694.1 (02/12), Figure I.3
        https://www.itu.int/rec/T-REC-G.694.1-201202-I/en

    >>> nval, mval = slots_to_m(6, 20)
    >>> nval
    13
    >>> mval
    7

    """
    nvalue = (int)((startn + stopn + 1) / 2)
    mvalue = (int)((stopn - startn + 1) / 2)
    return nvalue, mvalue


def m_to_freq(nvalue, mvalue, grid=0.00625e12):
    """ converts m into frequency range
        spectrum(13,7) is (193137500000000.0, 193225000000000.0)
        reference to Recommendation G.694.1 (02/12), Figure I.3
        https://www.itu.int/rec/T-REC-G.694.1-201202-I/en

    >>> fstart, fstop = m_to_freq(13, 7)
    >>> fstart
    193137500000000.0
    >>> fstop
    193225000000000.0

    """
    startn, stopn = mvalue_to_slots(nvalue, mvalue)
    fstart = nvalue_to_frequency(startn, grid)
    fstop = nvalue_to_frequency(stopn + 1, grid)
    return fstart, fstop