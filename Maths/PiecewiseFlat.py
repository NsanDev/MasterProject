from bisect import bisect_left

def piecewise_flat(t,p,times):
    assert (len(times)==len(p))
    assert (t<=times[-1]) #last element should be higher than t
    assert (all(times[i]<times[i+1] for i in range(0,len(times)-1)))
    if t < times[0]:
        return p[0]
    else:
        return p[bisect_left(times,t)]
