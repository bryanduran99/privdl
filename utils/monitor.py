class MonitorVar:
    '''Monitor 监控的变量'''

    def __init__(self, get, to_str=str):
        '''get: 获取变量值的函数，to_str: 将获取的变量值转化为字符串的函数'''
        self.get = get
        self.to_str = to_str

    def value(self):
        '''返回 get 得到的变量'''
        return self.get()

    def string(self):
        '''将 get 得到的变量用 to_str 转化为字符串后返回'''
        return self.to_str(self.get())

    def val_str(self):
        '''返回 get 得到的变量，和用 to_str 转化得到的字符串'''
        val = self.get()
        return val, self.to_str(val)


class Monitor:
    '''变量监控器\n
    monitor = Monitor() # 创建监控器\n
    monitor[key] = lambda: var # 添加一个监控变量\n
    monitor(key1=lambda: var1, key2=lambda: var2) # 添加多个监控变量 \n
    monitor.add(key=key, get=lambda: var, to_str=str)
    # 添加一个监控变量，且提供一个将变量转化为字符串的方法（便于 print）\n
    monitor[key] -> 变量的值 \n
    monitor.string(key) -> 变量的值的字符串形式 \n
    monitor.val_str(key) -> 变量的值 和 相应的字符串形式 \n
    monitor.check(*keys, Print=True) -> {key: val, ...} # 详见 doc'''

    def __init__(self):
        '''初始化监控变量字典为空'''
        self.vars = {}

    def __setitem__(self, key, get):
        self.add(key, get)

    def __call__(self, **kwargs):
        for key, get in kwargs.items():
            self.add(key, get)

    def add(self, key, get, to_str=str):
        '''添加监控的变量，key: 待监控变量名，
        get: 获取变量值的函数，to_str: 将获取的变量值转化为字符串的函数'''
        self.vars[key] = MonitorVar(get, to_str)

    def __getitem__(self, key):
        '''返回变量的值'''
        return self.vars[key].value()

    def string(self, key):
        '''返回变量的值的字符串形式'''
        return self.vars[key].string()

    def val_str(self, key):
        '''返回变量的值 和 相应的字符串形式'''
        return self.vars[key].val_str()

    def check(self, *keys, Print=True):
        '''获取监控变量的值, return {key: val, ...}\n
        默认会将这些变量的字符串形式 print 出来，若不想 print 请设置 Print=False \n
        check() -> 所有监控变量的值\n
        check('time', 'epoch', 'loss') -> 部分监控变量的值(按顺序)\n
        check('time', 'epoch', ..., 'loss') ->  所有监控变量的值(按顺序)'''
        # 根据 keys 的形式进行相应的补全
        if not keys: # check()
            keys = self.vars.keys()
        elif ... not in keys: # check('time', 'epoch', 'loss')
            pass
        else: # check('time', 'epoch', ..., 'loss')
            assert keys.count(...) == 1, '仅允许有一个省略号'
            missing_keys = set(self.vars) - set(keys)
            all_keys = []
            for key in keys:
                if key is not ...:
                    all_keys.append(key)
                else:
                    all_keys += list(missing_keys)
            keys = all_keys
        # 获取这些 key 对应的监控变量的值和其字符串形式
        vals, strs = {}, {}
        for key in keys:
            vals[key], strs[key] = self.val_str(key)
        if Print:
            print(' '.join(f'{key}={strs[key]}' for key in keys))
        return vals
