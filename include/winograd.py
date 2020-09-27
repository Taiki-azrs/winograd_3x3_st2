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

def f2x2_3x3(In,W):
    At=np.array([[1,1,1,0],[0,1,-1,1]])
    A=At.transpose()
    G=np.array([[1,0,0],[0.5,0.5,0.5],[0.5,-0.5,0.5],[0,0,1]])
    Gt=G.transpose()
    Bt=np.array([[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,-1,0,1]])
    B=Bt.transpose()
    #np.dot(np.dot(At,*)),A)
    ans1=np.dot(np.dot(G,W),Gt)
    ans2=np.dot(np.dot(Bt,In),B)
    ans3=np.dot(np.dot(At,ans1*ans2),A)
    return ans3

def f2x2_4x4(In,W):
    At=np.array([[1,1,1,1,0],[0,1,-1,2,1]])
    A=At.transpose()
    G=np.array([[0.5,0,0,0],
                [-0.5,-0.5,-0.5,-0.5],
                [-1.0/6,1.0/6,-1.0/6,1.0/6],
                [1.0/6,1.0/3,2.0/3,4/3],
                [0,0,0,1]])
    Gt=G.transpose()
    Bt=np.array([[2,-1,-2,1,0],
                 [0,-2,-1,1,0],
                 [0, 2,-3,1,0],
                 [0,-1,0,1,0],
                 [0, 2,-1,-2,1]])
    B=Bt.transpose()
    #ans=np.dot(np.dot(At,(np.dot(np.dot(G,W),Gt)*np.dot(np.dot(Bt,In),B))),A)
    ans1=np.dot(np.dot(G,W),Gt)
    ans2=np.dot(np.dot(Bt,In),B)
    ans3=np.dot(np.dot(At,ans1*ans2),A)
    return ans3
def f2x2_4x3(In,W):
    #f(2,4)
    AT=np.array([[1,1,1,1,0],[0,1,-1,2,1]])
    
    G=np.array([[  0.5,   0,   0,   0],
                [ -0.5,-0.5,-0.5,-0.5],
                [-1.0/6,1.0/6,-1.0/6,1.0/6],
                [1.0/6,1.0/3,2.0/3,4.0/3],
                [0,0,0,1]
    ])
    
    
    BT=np.array([[2,-1,-2,1,0],
                 [0,-2,-1,1,0],
                 [0,2,-3,1,0 ],
                 [0,-1,0,1,0 ],
                 [0,2,-1,-2,1]])
    
    #f(2,3) winograd
    
    AT3=np.array([[1,1,1,0],[0,1,-1,1]])
    A3=AT3.transpose()
    
    G3=np.array([[1,0,0],[0.5,0.5,0.5],[0.5,-0.5,0.5],[0,0,1]])
    GT3=G3.transpose()
    
    BT3=np.array([[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,-1,0,1]])
    B3=BT3.transpose()
    
    wans=np.dot(np.dot(G,W),GT3)*np.dot(np.dot(BT,In),B3)
    wans=np.dot(np.dot(AT,wans),A3)
    return wans
