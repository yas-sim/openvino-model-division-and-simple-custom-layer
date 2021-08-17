import numpy as np

dt = np.zeros((5,4,3), dtype=np.uint8)
print(dt.shape)
cnt = 0
for c in range(dt.shape[0]):
    for b in range(dt.shape[1]):
        for a in range(dt.shape[2]):
            dt[c,b,a] = cnt
            cnt += 1

print(dt)
print(dt.tobytes())
print(dt[0,0,0], dt[0,0,1], dt[0,0,2])
print(dt[0,0,0], dt[0,1,0], dt[0,2,0])
print(dt[0,0,0], dt[1,0,0], dt[2,0,0])
