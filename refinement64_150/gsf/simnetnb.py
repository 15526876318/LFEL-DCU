import math
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

seqdic={'A':0, 'R':1, 'D':2, 'C':3, 'Q':4, 'E':5, 'H':6, 'I':7, 'G':8, 'N':9, 'L':10, 'K':11, 'M':12, 'F':13, 'P':14, 'S':15, 'T':16, 'W':17, 'Y':18, 'V':19, 'X':20}
device =torch.device("cpu") 

class Sim_p(nn.Module):
    def __init__(self):
        super(Sim_p, self).__init__()
        #self.linear1 = nn.Linear(805, 512)
        self.linear1 = nn.Linear(770, 512)
        #self.linear1 = nn.Linear(1210, 512)
        #self.linear1 = nn.Linear(100, 512)
        self.relu1= nn.ReLU()
        self.linear2 = nn.Linear(512, 512)
        self.relu2= nn.ReLU()
        self.linear3 = nn.Linear(512, 512)
        self.relu3= nn.ReLU()
        self.linear4 = nn.Linear(512, 21)

    def forward(self, x):
        #x=self.relu1(self.gc1(x,adj))
        x=self.relu1(self.linear1(x))
        x=self.relu2(self.linear2(x))
        x=self.relu3(self.linear3(x))
        x=self.linear4(x)        
        return x

class NeRF_net(nn.Module):
    "对主链ca，c，n坐标变换"
    def __init__(self, inner_tensor,std = 10):
        super(NeRF_net,self).__init__()
        self.inner_tensor = inner_tensor
        self.std = std
        #self.mainchain_coord = mainchain_coord
        self.r_v = inner_tensor[:,0]
        self.angle_v = inner_tensor[:,1]
        self.dhd = inner_tensor[:,2]
        self.dhd_v = []
        for i in range(len(self.dhd)):
            t = self.dhd[i].view(1)
            t.requires_grad=True
            self.dhd_v.append(t)

        for i in range(int(len(self.dhd_v)/3)):
            self.dhd_v[i*3+1].requires_grad = False
         
    def forward(self, mainchain_coord):
        #dhd_v = torch.cat(self.dhd_v,0)
        self.new_c = []
        self.D2C = []
        if self.std:
            print(self.std)
            self.dhd = self.dhd+torch.rand_like(torch.Tensor(self.dhd))*self.std
        for i in range(3):
            self.new_c.append(mainchain_coord[i].view(1,-1))

        for i,coord in enumerate(self.inner_tensor):
            r, theta, phy = self.r_v[i], self.angle_v[i], self.dhd_v[i]
            x = r*torch.cos(theta).view(1,-1)
            y = r*torch.cos(phy)*torch.sin(theta).view(1,-1)
            z = r*torch.sin(phy)*torch.sin(theta).view(1,-1)
            D2 = torch.cat((x,y,z),0)
            #print(D2)
            A = self.new_c[i]
            B = self.new_c[i+1]
            C = self.new_c[i+2]
            vec1 = B - A
            vec2 = C - B
            vec1_n = torch.norm(vec1,2)
            vec2_n = torch.norm(vec2,2)
            bc =  vec2/vec2_n
            #print(bc.shape, vec1.shape)
            ab_bc = torch.cross(vec1,bc)
            ab_bc_n = torch.norm(ab_bc,2)
            n_n = ab_bc/ab_bc_n 
            M = torch.cat((bc, torch.cross(n_n, bc), n_n))
            M= torch.Tensor(M.reshape(3,3))
            M = torch.t(M)
            D = torch.t(torch.matmul(M,D2))+C
            #print(D)
            self.new_c.append(D)
            #self.D2C.append(D2)
        c = torch.cat(self.new_c)
        #d2c = torch.cat(self.D2C)
        #print(c.shape)
        return c

class NeRF_net_cb(nn.Module):
    def __init__(self, inner_coord_cb):
        super(NeRF_net_cb,self).__init__()
        self.inner_tensor = inner_coord_cb
        #self.mainchain_coord = mainchain_coord
         
    def forward(self, mainchain_coord):
        #dhd_v = torch.cat(self.dhd_v,0)
        self.new_c = []
        for i,coord in enumerate(self.inner_tensor):
            r, theta, phy = self.inner_tensor[i][0],self.inner_tensor[i][1], self.inner_tensor[i][2]
            x = r*torch.cos(theta).view(1,-1)
            y = r*torch.cos(phy)*torch.sin(theta).view(1,-1)
            z = r*torch.sin(phy)*torch.sin(theta).view(1,-1)
            D2 = torch.cat((x,y,z),0)
            #print(D2)
            A = mainchain_coord[3*i]
            B = mainchain_coord[3*i+1]
            C = mainchain_coord[3*i+2]
            vec1 = B - A
            vec2 = C - B
            vec1_n = torch.norm(vec1,2)
            vec2_n = torch.norm(vec2,2)
            bc =  vec2/vec2_n
            #print(bc.shape, vec1.shape)
            ab_bc = torch.cross(vec1,bc)
            ab_bc_n = torch.norm(ab_bc,2)
            n_n = ab_bc/ab_bc_n 
            M = torch.cat((bc, torch.cross(n_n, bc), n_n))
            M= torch.Tensor(M.reshape(3,3))
            M = torch.t(M)
            D = torch.t(torch.matmul(M,D2))+C
            #print(D)
            self.new_c.append(D)
        c = torch.cat(self.new_c)
        return c

class g_data_net(nn.Module):
    def __init__(self):
        super(g_data_net, self).__init__()

    def forward(self, mask, num_cs , dist, angle, seqlist):
        ids = num_cs
        seq = seqlist
        idx_t = []
        angle_t = []
        label = []
        dis_t =[]
        kk =0
        for i in range(len(mask)):
            idx_=[]
            dis_=[]
            angle_=[]
            for j in range(len(ids[i])):
                if len(idx_)==10:
                    break
                if abs(ids[i][j]-i)>6:
                    idx_.append(seqdic[seq[ids[i][j]]])
                    dis_ij = torch.unsqueeze(dist[i][j],0)
                    dis_.append(dis_ij)
                    angle_.append(angle[i][j])
            while len(idx_)<10:
                idx_.append(22)
                dis_.append(torch.zeros(1))
                angle_.append(torch.zeros(6))
            nb_id = [j for j in range(-6+i,7+i) if j!=i]
            for a in nb_id:
                if a in ids[i]:
                    k=np.where(ids[i]==a)
                    k=k[0][0] #duole yiwei suoyixuyao suoyin liangci
                    if seq[a] in seqdic:
                        idx_.append(seqdic[seq[a]])
                    else:
                        idx_.append(20)
                    dis_ik = torch.unsqueeze(dist[i][k],0)
                    dis_.append(dis_ik)
                    angle_.append(angle[i][k])
                else:
                    idx_.append(22)
                    dis_.append(torch.zeros(1))
                    angle_.append(torch.zeros(6))
            angle_ = torch.cat(angle_)
            dis_ = torch.cat(dis_)
            idx_t.append(idx_)
            dis_t.append(dis_)
            angle_t.append(angle_)
            if seq[i] in seqdic:
                label.append(seqdic[seq[i]])
            else:
                label.append(20)
            kk+=1
        data_t = torch.eye(23)
        dis_t = torch.cat(dis_t)/10
        angle_t = torch.cat(angle_t)/3
        idx_t =  torch.Tensor(idx_t).long()
        label = torch.Tensor(label).long()
        return data_t, idx_t, dis_t, angle_t, label, kk


class g_data_net_cpu(nn.Module):
    def __init__(self):
        super(g_data_net_cpu, self).__init__()

    def forward(self, mask, num_cs , dist, angle, seqlist):
        ids = num_cs
        seq = seqlist
        idx_t = []
        angle_t = []
        label = []
        dis_t =[]
        kk =0
        for i in range(len(mask)):
            idx_=[]
            dis_=[]
            angle_=[]
            for j in range(len(ids[i])):
                if len(idx_)==10:
                    break
                if abs(ids[i][j]-i)>6:
                    idx_.append(seqdic[seq[ids[i][j]]])
                    dis_ij = torch.unsqueeze(dist[i][j],0)
                    dis_.append(dis_ij)
                    angle_.append(angle[i][j])
            while len(idx_)<10:
                idx_.append(22)
                dis_.append(torch.zeros(1))
                angle_.append(torch.zeros(6))
            nb_id = [j for j in range(-6+i,7+i) if j!=i]
            for a in nb_id:
                if a in ids[i]:
                    k=np.where(ids[i]==a)
                    k=k[0][0] #duole yiwei suoyixuyao suoyin liangci
                    if seq[a] in seqdic:
                        idx_.append(seqdic[seq[a]])
                    else:
                        idx_.append(20)
                    dis_ik = torch.unsqueeze(dist[i][k],0)
                    dis_.append(dis_ik)
                    angle_.append(angle[i][k])
                else:
                    idx_.append(22)
                    dis_.append(torch.zeros(1))
                    angle_.append(torch.zeros(6))
            angle_ = torch.cat(angle_)
            dis_ = torch.cat(dis_)
            idx_t.append(idx_)
            dis_t.append(dis_)
            angle_t.append(angle_)
            if seq[i] in seqdic:
                label.append(seqdic[seq[i]])
            else:
                label.append(20)
            kk+=1
        data_t = torch.eye(23)
        dis_t = torch.cat(dis_t)/10
        angle_t = torch.cat(angle_t)/3
        idx_t =  torch.Tensor(idx_t).long()
        label = torch.Tensor(label).long()
        return data_t, idx_t, dis_t, angle_t, label, kk

class g_data_net_tian0(nn.Module):
    "特征整理成x"
    def __init__(self):
        super(g_data_net_tian0, self).__init__()

    def forward(self, dist, angle, idx_t, index_t, index_h):
        angle2_sin = torch.sin(angle)
        angle2_cos = torch.cos(angle)
        angle = torch.cat((angle2_sin, angle2_cos), 2)
        dist_00 = torch.zeros_like(dist[:,0]).unsqueeze(1) #增加一行0，用于补零
        angle_00 = torch.zeros_like(angle[:,0]).unsqueeze(1)
        dist_new = torch.cat((dist,dist_00), 1)
        angle_new = torch.cat((angle,angle_00),1)

        # print(idx_t.shape)
        h, w = idx_t.size()
        a = index_h.view(-1, 1).float()
        b = torch.ones(w,device = device).view(1, -1)
        ab = torch.mm(a, b).long()

        data_t = torch.eye(22,device = device)
        dist_t = dist_new[ab,index_t] #
        angle_t = angle_new[ab,index_t] #
        x1 = data_t[idx_t.view(-1)].view(h,-1)
        dist = dist_t.view(h, -1)
        angle = angle_t.view(h, -1)
        #angle_n = (angle-angle.min())/(angle.max() - angle.min())
        dist_n = (dist-dist.min())/(dist.max()-dist.min())
        angle_n = angle
        #dist_n = dist/10
        #print((angle-angle.min()).shape, dist.max()-dist.min())
        x=torch.cat((x1,dist_n,angle_n),1)
        #idx_t = torch.tensor(list(map(lambda x: seqdic[seq_list[x]],num_cs10.view(-1)))).view(-1,22)
        return x




    
