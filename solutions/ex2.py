import numpy as np

print(f"Exercise 2::")
print(f"  2.1:")
a  = np.fromfunction(lambda x,y: x+y,(5,5),dtype='int32')
print(f" a:\n{a}\n")

print(f"  2.2:")
# Solution 1:
b = np.eye(6,6) + np.eye(6,6,k=1) + np.eye(6,6,k=-1)
b = b.astype(dtype='bool')
print(f" b:\n{b}\n")

# Solution 2:
c = np.diag([1 for i in range(6)]) +\
    np.diag([1 for i in range(5)], k=1) +\
    np.diag([1 for i in range(5)],k=-1)
c = c.astype(dtype='bool')
print(f" c:\n{c}\n")    
