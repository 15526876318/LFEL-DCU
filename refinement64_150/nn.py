import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os 

class idxLayer(nn.Module):
    def __init__(self):
        super(idxLayer, self).__init__()

    def forward(self, x, idx, dis, angle_t):
        h, w = idx.size()
        x1 = x[idx.view(-1)].view(h,-1)
        dis_ = dis.view(h, -1)
        dis_ = (dis_-dis_.min())/(dis_.max()-dis_.min())
        angle = angle_t.view(h, -1)
        angle_sin = torch.sin(angle)
        angle_cos = torch.cos(angle)
        x=torch.cat((x1,dis_,angle_sin, angle_cos),1)
        return(x)

class Sim(nn.Module):
    def __init__(self):
        super(Sim, self).__init__()
        self.idxl=idxLayer()
        self.linear1 = nn.Linear(770, 512)
        self.relu1= nn.ReLU()
        self.linear2 = nn.Linear(512, 512)
        self.relu2= nn.ReLU()
        self.linear3 = nn.Linear(512, 512)
        self.relu3= nn.ReLU()
        self.linear4 = nn.Linear(512, 21)

    def forward(self, x,idx,dis,angle_t):
        x=self.idxl(x,idx,dis,angle_t)
        #print(x.shape)
        x=self.relu1(self.linear1(x))
        x=self.relu2(self.linear2(x))
        x=self.relu3(self.linear3(x))
        x=self.linear4(x)        
        return x

class Sim_p(nn.Module):
    def __init__(self):
        super(Sim_p, self).__init__()
        #self.linear1 = nn.Linear(805, 512)
        self.linear1 = nn.Linear(770, 512)
        self.relu1= nn.ReLU()
        self.linear2 = nn.Linear(512, 512)
        self.relu2= nn.ReLU()
        self.linear3 = nn.Linear(512, 512)
        self.relu3= nn.ReLU()
        self.linear4 = nn.Linear(512, 21)

    def forward(self, x):
        #x=self.relu1(self.gc1(x,adj))
        #print(x.dtype)
        x=self.relu1(self.linear1(x))

        x=self.relu2(self.linear2(x))
        x=self.relu3(self.linear3(x))
        x=self.linear4(x)        
        return x        
        
class NeRF_net(nn.Module):
    "将内坐标转为内坐标，初始化先将给内坐标中的两种二面角设置梯度。"
    def __init__(self, inner_tensor,std = 0.01):
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
        "forward完场计算图的前向传播，这里的mainchain的坐标需要用到前三个坐标作为起始"
        #dhd_v = torch.cat(self.dhd_v,0)
        self.new_c = []
        self.D2C = []
        #if self.std:
            #print(1)
            #self.dhd = self.dhd+torch.rand_like(torch.Tensor(self.dhd))*self.std
            #self.dhd = self.dhd+torch.Tensor(self.dhd)*self.std
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
            #print(bc.dtype,n_n.dtype, bc.dtype)
            M = torch.cat((bc, torch.cross(n_n, bc), n_n))
            #M= torch.Tensor(M.reshape(3,3))
            M= M.reshape(3,3)
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
            #M= torch.Tensor(M.reshape(3,3))
            M= M.reshape(3,3)
            M = torch.t(M)
            D = torch.t(torch.matmul(M,D2))+C
            #print(D)
            self.new_c.append(D)
        c = torch.cat(self.new_c)
        return c
        
class g_data_net_cpu(nn.Module):
    def __init__(self):
        super(g_data_net_cpu, self).__init__()

    def forward(self, mask, num_cs , dist, angle, seqlist,seqdic):
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
                idx_.append(21)
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
                    idx_.append(21)
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
        data_t = torch.eye(22)
        dis_t = torch.cat(dis_t)
        angle_t = torch.cat(angle_t)
        idx_t =  torch.Tensor(idx_t).long()
        label = torch.Tensor(label).long()
        return data_t, idx_t, dis_t, angle_t, label, kk        
        
class g_data_net_tian0(nn.Module):
    "特征整理成x"
    def __init__(self):
        super(g_data_net_tian0, self).__init__()

    def forward(self, dist, angle, idx_t, index_t, index_h,device):
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

class PNERF_net(nn.Module):
    def __init__(self, inner_tensor):
        super(PNERF_net,self).__init__()
        self.inner_tensor = inner_tensor
        self.bond_length = self.inner_tensor[:,0]
        self.bond_angle = self.inner_tensor[:,1]
        self.dhd = inner_tensor[:,2]
        #self.psi= torch.tensor(self.dhd[::3], requires_grad = True)
        self.psi = self.dhd[::3].clone().detach().requires_grad_(True)
        #self.phy = torch.tensor(self.dhd[2::3], requires_grad = True)
        self.phy = self.dhd[2::3].clone().detach().requires_grad_(True)
        self.omiga = inner_tensor[:,2][1::3]
        self.init_matrix = torch.tensor([[-np.sqrt(1.0 / 2.0), np.sqrt(3.0 / 2.0), 0],
                     [-np.sqrt(2.0), 0, 0], [0, 0, 0]],)
                    #dtype = torch.float64)###########################################################
    def cal_sfr_coord(self):
        #将原来算法串行的一部分，计算 special reference frame直接并行计算。
        self.dhd_v = torch.stack([self.psi , self.omiga,self.phy],dim= 1).reshape(len(self.inner_tensor))
        r_cos_theta = self.bond_length * np.cos(self.bond_angle)
        r_sin_theta = self.bond_length * np.sin(self.bond_angle)
        point_x = r_cos_theta
        point_y = torch.cos(self.dhd_v) * r_sin_theta
        point_z = torch.sin(self.dhd_v) * r_sin_theta
        point = torch.stack([point_x, point_y, point_z],dim =1)
        return point 
    
    def rotate_matrix(self, init_matrix, point):
        #根据起始坐标先进行一次旋转。
        new_c = []
        self.point=point
        for i in range(3):        
            new_c.append(init_matrix[i]) 
        for i in range(len(self.point)):
            A = new_c[i]
            B = new_c[i+1]
            C = new_c[i+2]
            vec1 = B - A
            vec2 = C - B
            #vec1_n = torch.norm(vec1,2)
            vec2_n = torch.norm(vec2,2)
            bc =  vec2/vec2_n
            #print(bc.shape, vec1.shape)
            ab_bc = torch.cross(vec1,bc)
            ab_bc_n = torch.norm(ab_bc,2)
            n_n = ab_bc/ab_bc_n 
            M = torch.cat((bc, torch.cross(n_n, bc), n_n))
            #M= torch.Tensor(M.reshape(3,3))
            M= M.reshape(3,3)
            M = torch.t(M)
            D2 = self.point[i]
            #print(M.dtype, D2.dtype, C.dtype)
            D = torch.t(torch.matmul(M,D2))+C
            new_c.append(D)
            #print("M:", M)
        new_c = torch.stack(new_c)
        return new_c 
    def srf2f(self, matrix, SRF_point):
        "计算最终坐标系下的坐标"
        new_c = []
        D2_list =SRF_point
        for i in range(3):        
            new_c.append(matrix[i]) 
        A = new_c[0]
        B = new_c[1]
        C = new_c[2]
        mk_1 = B - A
        mk = C - B 
        mk_m =  mk/torch.norm(mk,2)
        nk = torch.cross(mk_1,mk_m)
        nk_m = nk/torch.norm(nk,2)
        #xxxx = torch.cross(nk_m,mk_m)
        #print(mk_m.dtype, xxxx.dtype, nk_m.dtype)
        M = torch.cat((mk_m, torch.cross(nk_m , mk_m ), nk_m))
        #M= torch.Tensor(M.reshape(3,3))
        M= M.reshape(3,3)

        M = torch.t(M)
        for i in range(len(SRF_point)):
            D2 = D2_list[i] 
            #print(i,D2.dtype,C)       
            D = torch.t(torch.matmul(M,D2))+C
            
            new_c.append(D)
        new_c = torch.stack(new_c)
        return new_c
    def forward(self,fn,mainchain_coord_tensor):
        point = self.cal_sfr_coord()
        framment_l = int(len(point)/fn)  
        #print(framment_l)
        new_cc3 = []
        init_f =[]
        init_f.append(mainchain_coord_tensor[:3])
        for i in range(fn+1):              
            point_ = point[i*framment_l:(i+1)*framment_l]
            new_c1 = self.rotate_matrix(self.init_matrix, point_)
            #print(point_.shape, new_c1.shape)
            new_cc1 = self.srf2f(init_f[i], new_c1[3:])
            init_f.append(new_cc1[-3:])
            if i>0:
                new_cc3.append(new_cc1[3:])
            else:
                new_cc3.append(new_cc1)  
        #cccc = torch.cat([new_cc3[0],new_cc3[1][3:],new_cc3[2][3:] ])
        cccc = torch.cat(new_cc3) 
        return cccc
