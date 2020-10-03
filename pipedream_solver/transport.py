try:
    import numba
    _HAS_NUMBA = True
except:
    _HAS_NUMBA = False
if _HAS_NUMBA:
    from pipedream_solver.nquality import QualityBuilder
else:
    raise ImportError('Requires `numba` library.')
