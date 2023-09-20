import time 

def timer(fn):
    """source:https://www.codesansar.com/python-programming-examples/decorator-measure-elapsed-time.htm"""
    
    def inner(*args, **kwargs):
        t_start = time.time();
        result = fn(*args, **kwargs);
        t_elapsed = (time.time() - t_start)/60;
        print(f"Time elapsed for {fn.__name__} = {t_elapsed:.2f} minutes");
        return result;
    
    return inner;