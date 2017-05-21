from numpy import array_equal

from Scripts.parameters import load_array

V0 = load_array('contract_ini', folder='')
cva = load_array('cva_hull', folder='')
cva0 = load_array('cva_hull0', folder='')
bl = array_equal(cva, cva0)
a = 1
