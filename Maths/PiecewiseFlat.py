from bisect import bisect_left

'''
Create piece function from p and times and evaluate it at t.
the piecewise function is left continuous and defined on (-Inf;times[-1]] which is assumed to be sorted.
'''
def piecewise_flat(t,p,times):
    assert (len(times)==len(p))
    assert (t<=times[-1]) #last element should be higher than t
    assert (all(times[i]<times[i+1] for i in range(0,len(times)-1)))
    return p[bisect_left(times,t)]
