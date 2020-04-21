import numpy as np

#F(2x2,2x2)
def f2x2_2x2(In,W):
    G=np.array([[1,0],[1,1],[0,1]])
    GT=G.transpose()
    AT=np.array([[1,1,0],[0,1,1]])
    A=AT.transpose()
    BT=np.array([[1,-1,0],[0,1,0],[0,-1,1]])
    B=BT.transpose()
    wp=np.dot(np.dot(G,W),GT)
    pre=np.dot(np.dot(BT,In),B)
    conv=wp*pre
    post=np.dot(np.dot(AT,conv),A)
    return post
