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
    At=np.array([[1,1, 1,0],
                 [0,1,-1,1]])
    A=At.transpose()
    G=np.array([[  1,   0,  0],
                [0.5, 0.5,0.5],
                [0.5,-0.5,0.5],
                [  0,   0,  1]])
    Gt=G.transpose()
    Bt=np.array([[1, 0,-1,0],
                 [0, 1, 1,0],
                 [0,-1, 1,0],
                 [0,-1, 0,1]])
    B=Bt.transpose()
    ans1=np.dot(np.dot(G,W),Gt)
    ans2=np.dot(np.dot(Bt,In),B)
    ans3=np.dot(np.dot(At,ans1*ans2),A)
    return ans3

def f2x2_4x4(In,W):
    At=np.array([[1,1, 1,1,0],
                 [0,1,-1,2,1]])
    A=At.transpose()
    G=np.array([[   0.5,    0,     0,    0],
                [  -0.5, -0.5,  -0.5, -0.5],
                [-1.0/6,1.0/6,-1.0/6,1.0/6],
                [1.0/6,1.0/3,2.0/3,4/3],
                [0,0,0,1]])
    Gt=G.transpose()
    Bt=np.array([[2,-1,-2 ,1,0],
                 [0,-2,-1 ,1,0],
                 [0, 2,-3 ,1,0],
                 [0,-1, 0 ,1,0],
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
    
    
    BT=np.array([[2,-1,-2, 1,0],
                 [0,-2,-1, 1,0],
                 [0, 2,-3, 1,0],
                 [0,-1, 0, 1,0],
                 [0, 2,-1,-2,1]])
    
    #f(2,3) winograd
    BT3=np.array([[1, 0,-1,0],
                  [0, 1, 1,0],
                  [0,-1, 1,0],
                  [0,-1, 0,1]])
    AT3=np.array([[1,1,1,0],[0,1,-1,1]])
    A3=AT3.transpose()
    
    G3=np.array([[1,0,0],[0.5,0.5,0.5],[0.5,-0.5,0.5],[0,0,1]])
    GT3=G3.transpose()
    
 
    B3=BT3.transpose()
    
    wans=np.dot(np.dot(G,W),GT3)*np.dot(np.dot(BT,In),B3)
    wans=np.dot(np.dot(AT,wans),A3)
    return wans
def conver_w(W):
    tmp=np.zeros((9,7))
    tmp2=np.zeros((9,9))
    dW=np.zeros((9,9))
    g1=np.array([[   0.5,    0,     0,    0],
                [  -0.5, -0.5,  -0.5, -0.5],
                [-1.0/6,1.0/6,-1.0/6,1.0/6],
                [1.0/6,1.0/3,2.0/3,4/3],
                [0,0,0,1]])
    g1t=g1.transpose()
    g2=np.array([[1,0,0],[0.5,0.5,0.5],
                 [0.5,-0.5,0.5],[0,0,1]])
    g2t=g2.transpose()

    g4=np.array([[  1,   0,  0],
                [0.5, 0.5,0.5],
                [0.5,-0.5,0.5],
                [  0,   0,  1]])
    g4t=g4.transpose()
    for i in range(9):
        if(i%2==0):
            for k in range(4):
                tmp[i,:]+=W[k*2,:]*g1[int(i/2),k]
        else:
            for k in range(3):
                tmp[i,:]+=W[k*2+1,:]*g2[int(i/2),k]

    for j in range(9):
        if(j%2==0):
            for k in range(4):
                tmp2[:,j]+=tmp[:,k*2]*g1[int(j/2),k]
        else:
            for k in range(3):
                tmp2[:,j]+=tmp[:,k*2+1]*g2[int(j/2),k]
            """
    tmp[0,:]=W[0,:]*0.5;
    tmp[2,:]=W[0,:]*-0.5+W[2,:]*-0.5+W[4,:]*-0.5+W[6,:]*-0.5
    tmp[4,:]=W[0,:]*g1[2,0]+W[2,:]*g1[2,1]+W[4,:]*g1[2,2]+W[6,:]*g1[2,3]
    tmp[6,:]=W[0,:]*g1[3,0]+W[2,:]*g1[3,1]+W[4,:]*g1[3,2]+W[6,:]*g1[3,3]
    tmp[8,:]=W[0,:]*g1[4,0]+W[2,:]*g1[4,1]+W[4,:]*g1[4,2]+W[6,:]*g1[4,3]

    tmp[1,:]=W[1,:]
    tmp[3,:]=W[1,:]*g2[1,0]+W[3,:]*g2[1,1]+W[5,:]*g2[1,2]
    tmp[5,:]=W[1,:]*g2[2,0]+W[3,:]*g2[2,1]+W[5,:]*g2[2,2]
    tmp[7,:]=W[1,:]*g2[3,0]+W[3,:]*g2[3,1]+W[5,:]*g2[3,2]
    tmp2[:,0]=tmp[:,0]*0.5;
    tmp2[:,2]=tmp[:,0]*g1[1,0]+tmp[:,2]*g1[1,1]+tmp[:,4]*g1[1,2]+tmp[:,6]*g1[1,3]
    tmp2[:,4]=tmp[:,0]*g1[2,0]+tmp[:,2]*g1[2,1]+tmp[:,4]*g1[2,2]+tmp[:,6]*g1[2,3]
    tmp2[:,6]=tmp[:,0]*g1[3,0]+tmp[:,2]*g1[3,1]+tmp[:,4]*g1[3,2]+tmp[:,6]*g1[3,3]
    tmp2[:,8]=tmp[:,0]*g1[4,0]+tmp[:,2]*g1[4,1]+tmp[:,4]*g1[4,2]+tmp[:,6]*g1[4,3]

    tmp2[:,1]=tmp[:,1]
    tmp2[:,3]=tmp[:,1]*g2[1,0]+tmp[:,3]*g2[1,1]+tmp[:,5]*g2[1,2]
    tmp2[:,5]=tmp[:,1]*g2[2,0]+tmp[:,3]*g2[2,1]+tmp[:,5]*g2[2,2]
    tmp2[:,7]=tmp[:,1]*g2[3,0]+tmp[:,3]*g2[3,1]+tmp[:,5]*g2[3,2]
    dW[::2,::2]=np.dot(np.dot(g1,W[::2,::2]),g1t)
    dW[::2,1::2]=np.dot(np.dot(g1,W[::2,1::2]),g2t)
    dW[1::2,::2]=np.dot(np.dot(g2,W[1::2,::2]),g1t)
    dW[1::2,1::2]=np.dot(np.dot(g4,W[1::2,1::2]),g4t)
    """
    return tmp2
