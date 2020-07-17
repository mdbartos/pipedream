try:
    import numba
    _HAS_NUMBA = True
except:
    _HAS_NUMBA = False
if _HAS_NUMBA:
    from pipedream_solver.nsuperlink import nSuperLink as SuperLink
else:
    from pipedream_solver.superlink import SuperLink as SuperLink
