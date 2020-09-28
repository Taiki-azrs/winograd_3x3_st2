import numpy as np
import include.winograd as wino
ow=2;oh=2
iw=9;ih=9
st=2

#In=np.random.rand(ih,iw)
#W=np.random.rand(7,7)
In=np.arange(ih*iw).reshape(ih,iw)
W=np.arange(7*7).reshape(7,7)




#direct
ans=np.zeros((ow,oh))
tmp=np.zeros((ih,iw))
tmp2=np.zeros((ih,iw))
for i in range(oh):
    for j in range(ow):
        idy=i*st
        idx=j*st
        for r in range(7):
            for m in range(7):
                ans[i][j]+=In[idy+r][idx+m]*W[r][m]
print("Direct:\n",ans)

#winograd simple
"""
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
"""

#SIMD like winograd

#weight conversion 
dW=wino.conver_w(W)

#pre process
#dot(G,In)
tmp[0,:]=(In[0,:]-In[4,:])*2+In[6,:]-In[2,:]
tmp[2,:]=(In[6,:]-In[4,:])-In[2,:]*2
tmp[4,:]=In[2,:]*2-In[4,:]*3+In[6,:]
tmp[6,:]=In[6,:]-In[2,:]
tmp[8,:]=(In[2,:]-In[6,:])*2+In[8,:]-In[4,:]

tmp[1,:]=In[1,:]-In[5,:]
tmp[3,:]=In[3,:]+In[5,:]
tmp[5,:]=In[5,:]-In[3,:]
tmp[7,:]=In[7,:]-In[3,:]

#dot(tmp,GT)
tmp2[:,0]=(tmp[:,0]-  tmp[:,4])*2+tmp[:,6]-tmp[:,2]
tmp2[:,2]=(tmp[:,6]-  tmp[:,4])-  tmp[:,2]*2      
tmp2[:,4]= tmp[:,2]*2-tmp[:,4]*3+ tmp[:,6]        
tmp2[:,6]= tmp[:,6]-  tmp[:,2]                    
tmp2[:,8]=(tmp[:,2]-  tmp[:,6])*2+tmp[:,8]-tmp[:,4]

tmp2[:,1]=tmp[:,1]-tmp[:,5]
tmp2[:,3]=tmp[:,3]+tmp[:,5]
tmp2[:,5]=tmp[:,5]-tmp[:,3]
tmp2[:,7]=tmp[:,7]-tmp[:,3]

#convolution process
tmp2=tmp2*dW

ptmp=np.zeros((4,9))
ptmp2=np.zeros((4,4))

#post process
#dot(AT,tmp2)
ptmp[0,:]=tmp2[0,:]+tmp2[2,:]+tmp2[4,:]  +tmp2[6,:]
ptmp[2,:]=tmp2[2,:]-tmp2[4,:]+tmp2[6,:]*2+tmp2[8,:]

ptmp[1,:]=tmp2[1,:]+tmp2[3,:]+tmp2[5,:]
ptmp[3,:]=tmp2[3,:]-tmp2[5,:]+tmp2[7,:]

#dot(ptmp,A)
ptmp2[:,0]=ptmp[:,0]+ptmp[:,2]+ptmp[:,4]  +ptmp[:,6]
ptmp2[:,2]=ptmp[:,2]-ptmp[:,4]+ptmp[:,6]*2+ptmp[:,8]
                                       
ptmp2[:,1]=ptmp[:,1]+ptmp[:,3]+ptmp[:,5]
ptmp2[:,3]=ptmp[:,3]-ptmp[:,5]+ptmp[:,7]

wans=ptmp2[::2,::2]+ptmp2[1::2,::2]+ptmp2[::2,1::2]+ptmp2[1::2,1::2]

print("winograd:\n",wans)
print("error:\n",abs(ans-wans))
