class Box:
    '''make dict items accessable by "."'''
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def __repr__(self):
        return f'Box({self.__dict__})'