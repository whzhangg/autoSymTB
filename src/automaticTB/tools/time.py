import functools, time 

def timefn(fn):
    """this function print the execution time in second for a given function"""
    @functools.wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print(f"@timefn: {fn.__name__} took {t2 - t1:>.8f} seconds")
        return result

    return measure_time