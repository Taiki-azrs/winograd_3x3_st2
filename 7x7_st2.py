import numpy as np
import include.winograd as wino
ow=2;oh=2
iw=9;ih=9
st=2

In=np.random.rand(ih,iw)
W=np.random.rand(7,7)
#In=np.arange(ih*iw).reshape(ih,iw)
#W=np.arange(7*7).reshape(7,7)




#direct
ans=np.zeros((ow,oh))
for i in range(oh):
    for j in range(ow):
        idy=i*st
        idx=j*st
        for r in range(7):
            for m in range(7):
                ans[i][j]+=In[idy+r][idx+m]*W[r][m]
print("Direct:\n",ans)

#winograd
#1st
in_1st=In[::2,::2]
w_1st=W[::2,::2]
ans_1st=wino.f2x2_4x4(in_1st,w_1st)

#2nd
in_2nd=In[::2,1::2]
w_2nd=W[::2,1::2]
ans_2nd=wino.f2x2_4x3(in_2nd,w_2nd)

#3rd
in_3rd=In[1::2,::2]
w_3rd=W[1::2,::2]
ans_3rd=wino.f2x2_4x3(in_3rd.transpose(),w_3rd.transpose())
ans_3rd=ans_3rd.transpose()

#4th
in_4th=In[1::2,1::2]
w_4th=W[1::2,1::2]
ans_4th=wino.f2x2_3x3(in_4th,w_4th)

wans=ans_1st+ans_2nd+ans_3rd+ans_4th
print("winograd:\n",wans)
print("error:\n",abs(ans-wans))
