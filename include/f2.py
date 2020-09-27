import numpy as np

#f(2,2)
def f2_2(In,W):
    G=np.array([[1,0],[1,1],[0,1]])
    AT=np.array([[1,1,0],[0,1,1]])
    BT=np.array([[1,-1,0],[0,1,0],[0,-1,1]])
    wp=np.dot(G,W)
    pre=np.dot(BT,In)
    conv=wp*pre
    post=np.dot(AT,conv)
    return post
