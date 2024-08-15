class BaseCallback():
    def __init__(self, *args, **kwargs):
        pass

    def __on_step_start__(self, *args, **kwargs):
        return None
    
    def __on_step_end__(self, *args, **kwargs):
        return None

    def __on_save_state__(self, *args, **kwargs):
        return None

    def __on_load_state__(self, *args, **kwargs):
        return None

    def __on_simulation_start__(self, *args, **kwargs):
        return None

    def __on_simulation_end__(self, *args, **kwargs):
        return None
 