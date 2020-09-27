import numpy as np
def f2x2_3x3(In,W):
    At=np.array([[1,1,1,0],[0,1,-1,1]])
    A=At.transpose()
    G=np.array([[1,0,0],[0.5,0.5,0.5],[0.5,-0.5,0.5],[0,0,1]])
    Gt=G.transpose()
    Bt=np.array([[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,-1,0,1]])
    B=Bt.transpose()
    ans=np.dot(np.dot(At,(np.dot(np.dot(G,W),Gt)*np.dot(np.dot(Bt,In),B))),A)
    return ans
