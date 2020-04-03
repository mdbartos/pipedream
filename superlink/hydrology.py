try:
    import numba
    _HAS_NUMBA = True
except:
    _HAS_NUMBA = False
if _HAS_NUMBA:
    from superlink.ninfiltration import nGreenAmpt as GreenAmpt
else:
    from superlink.infiltration import GreenAmpt as GreenAmpt
