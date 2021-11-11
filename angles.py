import numpy as np

# x,y,z co-ordinates of points a, b, c
# At start, we assume that z coordinate is unknown, but defaut to 0
pt_a = np.array([0,0,0])
pt_b = np.array([1,2,0])
pt_c = np.array([3,4,0])

# calibrated lengths of segments ab and bc
len_ab = 5
len_bc = 5

# assign easy to "read" coordinates based on arrays
xa = pt_a[0]
ya = pt_a[1]
za = pt_a[2]

xb = pt_b[0]
yb = pt_b[1]
zb = pt_b[2]

# calculate the z position of point b
zb = np.sqrt(len_ab**2 - (xb-xa)**2 - (yb-ya)**2)

# assign easy to "read" coordinates based on array
xc = pt_c[0]
yc = pt_c[1]
zc = pt_c[2]

# calculate the z position of point b
zc = np.sqrt(len_bc**2 - (xc-xb)**2 - (yc-yb)**2)

# calculate the length of segment ac
len_ac = np.sqrt(xc**2 + yc**2 + zc**2)

# compute the angle between segments ab and bc
theta = np.arccos( (len_ab**2+len_bc**2-len_ac**2) / (2*len_ab*len_bc))
theta = theta*180/np.pi

print(theta)
