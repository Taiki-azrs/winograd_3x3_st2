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
#2st
in_2st=In[::2,1::2]
w_2st=W[::2,1::2]
ans2=f2.f2_2(in_2st,w_2st)
#3st
in_3st=In[1::2,::2]
w_3st=W[1::2,::2]
in_3st_t=in_3st.transpose()
w_3st_t=w_3st.transpose()
ans3=f2.f2_2(in_3st_t,w_3st_t).transpose()
#4st
in_4st=In[1::2,1::2]
w_4st=W[1::2,1::2]
ans4=in_4st*w_4st

wans=ans1+ans2+ans3+ans4
print("winograd:\n",wans)
print("error:\n",ans-wans)
