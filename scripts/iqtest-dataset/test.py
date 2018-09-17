import numpy as np

a = [
    [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ],

    [
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1]
    ],

    [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]

]

b= np.array(a)
print (b.shape)

c = b.flatten()
print(c)

d = "".join([str(x) for x in c])
print (d)

e = [int(x) for x in d]
print (e)

f = np.array(e).reshape(b.shape)
print (f.shape,f)

