try:
    import numba
    _HAS_NUMBA = True
except:
    _HAS_NUMBA = False
if _HAS_NUMBA:
    from superlink.numbalink import NumbaLink as SuperLink
else:
    from superlink.superlink import SuperLink as SuperLink
