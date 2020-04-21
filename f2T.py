import numpy as np
"""
In=np.arange(9)
In=In.reshape([3,3])
W=np.arange(4)
W=W.reshape([2,2])
ans=np.zeros((2,2))
for i in range(2):
    for j in range(2):
        for r in range(2):
            for m in range(2):
                ans[i][j]+=In[i+r][j+m]*W[r][m]

print("In:\n",In)
print("W:\n",W)
print("ans:\n",ans)
"""
def f2_2(In,W):
    G=np.array([[1,0],[1,1],[0,1]])
    AT=np.array([[1,1,0],[0,1,1]])
    BT=np.array([[1,-1,0],[0,1,0],[0,-1,1]])
    wp=np.dot(G,W)
    pre=np.dot(BT,In)
    conv=wp*pre
    post=np.dot(AT,conv)
    return post
