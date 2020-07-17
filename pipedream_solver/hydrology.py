try:
    import numba
    _HAS_NUMBA = True
except:
    _HAS_NUMBA = False
if _HAS_NUMBA:
    from pipedream_solver.ninfiltration import nGreenAmpt as GreenAmpt
else:
    from pipedream_solver.infiltration import GreenAmpt as GreenAmpt
