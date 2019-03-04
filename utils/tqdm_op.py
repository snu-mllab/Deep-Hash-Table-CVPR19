from tqdm import tqdm

def tqdm_range(*args, **kwargs):
    return tqdm(range(*args, **kwargs), ascii=True, desc="batch")

