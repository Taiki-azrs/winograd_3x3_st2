import numpy as np
import f2x2
import f2
#winograd_3x3_stride2

ow=2;oh=2 #output_size
iw=5;ih=5 #input_size 
st=2      #stride


In=np.random.rand(ih,iw)
W=np.random.rand(3,3)


#Direct
ans=np.zeros((ow,oh))
for i in range(0,iw-1,st):
    for j in range(0,ih-1,st):
        for r in range(3):
            for m in range(3):
                ans[int(i/st)][int(j/st)]+=In[i+r][j+m]*W[r][m]
print("Direct:\n",ans)


#winograd
#1st
in_1st=In[::2,::2]
w_1st=W[::2,::2]
ans1=f2x2.f2x2_2x2(in_1st,w_1st)
#2nd
in_2nd=In[::2,1::2]
w_2nd=W[::2,1::2]
ans2=f2.f2_2(in_2nd,w_2nd)
#3rd
in_3rd=In[1::2,::2]
w_3rd=W[1::2,::2]
in_3rd_t=in_3rd.transpose()
w_3rd_t=w_3rd.transpose()
ans3=f2.f2_2(in_3rd_t,w_3rd_t).transpose()
#4th
in_4th=In[1::2,1::2]
w_4th=W[1::2,1::2]
ans4=in_4th*w_4th

wans=ans1+ans2+ans3+ans4
print("winograd:\n",wans)
print("error:\n",ans-wans)
