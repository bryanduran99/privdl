class State:
    '''
    provide state_dict and load_state_dict methods for any obj,
    even if the obj is immutable.

    stated_obj = State(obj)

    stated_obj() -> obj

    stated_obj.state_dict() -> obj

    stated_obj.load_state_dict(another_obj)

    stated_obj() -> another_obj
    '''

    def __init__(self, obj):
        self.obj = obj
    
    def __call__(self):
        return self.obj
    
    def state_dict(self):
        return self.obj
    
    def load_state_dict(self, obj):
        self.obj = obj
