import numpy as np 

def get_modified_otsu_threshold(
    x1: np.ndarray, 
    x2: np.ndarray, 
    w1: float, 
    w2: float, 
    stepsize: int = 100,
) -> float :

    def __get_otsu_criterion(x1, x2, w1, w2, th) :
        x1 = x1[x1 < th]
        x2 = x2[x2>=th]
        var1 = np.var(x1) if x1.size>0 else 0
        var2 = np.var(x2) if x2.size>0 else 0
        
        return w1 * var1 + w2 * var2

    def __get_search_range(x1, x2) :
        m1, m2 = np.mean(x1), np.mean(x2)
        assert m1 < m2
        s1, s2 = np.std(x1), np.std(x2)
        lb, ub = m1+s1, m2-s2
        while lb >= ub :               
            s1 *= 0.95
            s2 *= 0.95
            lb, ub = m1+s1, m2-s2
        return lb, ub
              
    lb, ub = __get_search_range(x1, x2)
    th_range = np.linspace(lb, ub, stepsize)
    criteria = [__get_otsu_criterion(x1, x2, w1, w2, t) for t in th_range]
    th = th_range[np.argmin(criteria)]
    return th


def get_min_weighted_threshold(
    x1: np.ndarray, 
    x2: np.ndarray, 
    w1: float, 
    w2: float, 
    n_bins: int = 100,
) -> float :

    m1, m2 = np.mean(x1), np.mean(x2)
    assert m1 < m2
    x1, x2 = x1[(x1>m1) & (x1<m2)], x2[(x2>m1) & (x2<m2)]

    step = (m2 - m1) / n_bins
    x1 = ((x1 - m1) / step).astype(np.int32)
    x2 = ((x2 - m1) / step).astype(np.int32)     

    count = np.zeros((n_bins, ))
    a1, a2 = 1, x1.size / x2.size

    for x in x1 :
        count[x] += a1
    for x in x2 :
        count[x] += a2

    th = m1 + step * np.argmin(count[:-1])

    return th


def get_in_dist_threshold(
    x1: np.ndarray, 
    x2: np.ndarray, 
    w1: float, 
    w2: float, 
    n_std: float = 1.0,
) -> float :

    m1, m2 = np.mean(x1), np.mean(x2)
    assert m1 < m2
    s1, s2 = np.std(x1), np.std(x2)

    th = m1 + n_std * s1

    return th