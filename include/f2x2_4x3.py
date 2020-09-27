import numpy as np

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
