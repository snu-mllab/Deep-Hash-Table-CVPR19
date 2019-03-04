import numpy as np
import os

def params2id(*args):
    nargs = len(args)
    id_ = '{}'+'_{}'*(nargs-1)
    return id_.format(*args)

def listformat(list_, replace=False, src='0.', dst='.'):
    if replace:
        return '/'.join([str(v) for v in list_]).replace(src, dst)
    else:
        return "/".join([str(v) for v in list_])

def listformat_form(list_, form):
    return "/".join([form%v for v in list_])

class FileIdManager:
    def __init__(self, *attrs):
        self.attrs = attrs[0]
        self.nattr = len(self.attrs)
        #for attr in self.attrs: assert type(attr)==str, "Type of attributes should be set as string" 

    def get_id_from_args(self, args):
        tmp = list()
        for attr in self.attrs:
            if attr == '*': tmp.append('*')
            elif type(attr)!=str: tmp.append(attr)
            else: tmp.append(getattr(args, attr))
        return params2id(*tuple(tmp))

    def update_args_with_id(self, args, id_):
        id_split = id_.split('_')
        assert len(id_split)==self.nattr, "id_ should be composed of the same number of attributes"

        for idx in range(self.nattr):
            attr = self.attrs[idx]
            type_attr = type(getattr(args, attr))
            setattr(args, attr, type_attr(id_split[idx]))

if __name__=='__main__':
    a = [0.1, 0.2, 1.00]
    print(listformat(a, replace=True, src='0.', dst='.'))
