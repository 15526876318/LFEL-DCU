"read pdb file and prepare feature"
import numpy as np
import math
import random
import gsf.math_p as mp
from Bio.PDB.PDBParser import PDBParser
from Bio import PDB
#from simnetnb import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

seqdic={'A':0, 'R':1, 'D':2, 'C':3, 'Q':4, 'E':5, 'H':6, 'I':7, 'G':8, 'N':9, 'L':10, 'K':11, 
        'M':12, 'F':13, 'P':14, 'S':15, 'T':16, 'W':17, 'Y':18, 'V':19}
t_dic={'ALA':'A','VAL':'V','LEU':'L','ILE':'I','PHE':'F','TRP':'W','MET':'M','PRO':'P',\
       'GLY':'G','SER':'S','THR':'T','CYS':'C','TYR':'Y','ASN':'N','GLN':'Q','HIS':'H',\
       'LYS':'K','ARG':'R','ASP':'D','GLU':'E'}


class Read_pdb_feature(object):
    def __init__(self):
        self.aa_list_full = None
        self.ca_list = []
        self.cb_list = []
        self.c_list = []
        self.n_list = []
        self.mainchain = []
        self.seq_list = ''
        self.mask = []
        self.inner = []
        self.inner_cb = []
        self.cb_coord= []
        
    def readpdb(self, pdb_id, chain=None):
        p = PDBParser(PERMISSIVE=1)
        s = p.get_structure("pdb_s", pdb_id)
        try:
            s = s[0][chain]
        except:
            s= s
        self.res_list = PDB.Selection.unfold_entities(s, 'R')  #read aminoacid        
        self._check_res_atom()
        self._get_inner_coord()
        self._get_inner_coord_cb()
    
    def get_aa_list_full(self):
        return self.aa_list_full    
    
    def get_backbone(self):
        return self.mainchain

    def get_seq(self):
        return self.seq_list               

    def get_mask(self):
        return self.mask
    
    def get_inner_coord(self):
        return self.inner_coord
    
    def get_inner_coord_cb(self):
        return self.inner_coord_cb
    
    def mc_tensor(self):
        self.mainchain_coord_tensor = torch.Tensor([i.coord for i in self.mainchain])
        return self.mainchain_coord_tensor
    def get_cb_tensor(self):
        return self.cb_tensor
    
    #private
    def _check_res_atom(self):
        self.aa_list_full = [a for a in self.res_list if PDB.is_aa(a)]
        self._check_atom()
        self._check_cb()
        self._get_cb_tensor()

        
        
    def _check_atom(self): 
        for i, a in enumerate(self.aa_list_full):            
            if a.has_id('CA') and a.has_id('C') and a.has_id('N'):
                ca = a['CA']
                c = a['C']
                n = a['N'] 
                self.ca_list.append(ca)
                self.c_list.append(c)
                self.n_list.append(n)
                self.mainchain.append(n)
                self.mainchain.append(ca)
                self.mainchain.append(c)
            else:
                print("res %d do not have ca or c or n atom " % (i+self.aa_list_full[0].id[1]))
                
            try:
                t=a.get_resname()
                if t in t_dic:
                    self.seq_list+=t_dic[t]
                else:
                    self.seq_list+='X'
            except:
                self.seq_list+='X'
                
            if a.has_id('CA'):
                self.mask.append(1)
            else:
                self.mask.append(0)
                
    
    def _check_cb(self):
        for a in self.ca_list:
            try:
                t = a.parent['CB']
                self.cb_list.append(t)
            except:
                self.cb_list.append(None)  
                
    def _get_cb_tensor(self):
        for i in range(len(self.cb_list)):
            if self.cb_list[i]==None:
                ca_v=self.ca_list[i].get_vector().get_array()
                c_v=self.c_list[i].get_vector().get_array()
                n_v=self.n_list[i].get_vector().get_array()
                cb= self._calha1(n_v,c_v,ca_v)
                #cb=PDB.vectors.Vector(cb)
                atom= cb
            else:
                atom = self.cb_list[i].coord  
                
            self.cb_coord.append(atom)
            self.cb_tensor = torch.Tensor(self.cb_coord)
            
    def _calha1(self, a,b,c):
        "calculate gly H coord"
        ab=b-a
        cb=b-c
        bc=c-b
        cbmo=np.linalg.norm(cb)
        d=cb*1.0814/cbmo
        bcmo=np.linalg.norm(cb)
        bc/=bcmo
        fabc=np.cross(ab,cb)
        fmo=np.linalg.norm(fabc)
        fabc/=fmo
        d=self._rotation(d,fabc,math.pi*108.0300/180.0)
        d=self._rotation(d,bc,math.pi*117.8600/180.0)
        d+=c
        return d
    def _rotation(self,r,v,theta):
        t1=r*np.cos(theta)
        t2=np.cross(v,r)
        t2=t2*np.sin(theta)
        vr=np.dot(v,r)
        t3=vr*v*(1-np.cos(theta))
        r=t1+t2+t3
        return r
            
    def _get_inner_coord(self):    
        for i in range(3,len(self.mainchain)):
            atom = self.mainchain[i].coord
            #print(mainchain[i].id,atom)
            nb1 = self.mainchain[i-1].coord
            nb2 = self.mainchain[i-2].coord
            nb3 = self.mainchain[i-3].coord
            vec1 = nb2- nb3
            vec2 = nb1- nb2
            vec3 = atom- nb1
            #print(vec1.dtype,vec2.dtype,vec3.dtype)
            bond = mp.L_MO_ab(atom, nb1)
            #print(bond.dtype)
            angle = mp.get_angle(vec2, vec3)
            #print("angle",angle.dtype)
            dhd = mp.get_dhd(vec1, vec2, vec3)
            #print(dhd.dtype)
            inner_coord = bond, angle, dhd
            #print(inner_coord)
            self.inner.append(inner_coord)
        self.inner_coord = torch.from_numpy(np.array(self.inner,dtype= 'float32'))
    
    def _get_inner_coord_cb(self):
        for i in range(len(self.cb_list)):
            nb3 = self.n_list[i].coord
            nb2 = self.ca_list[i].coord
            nb1 = self.c_list[i].coord
            #if cb_list[i]==None and c_list[i]!=None and n_list[i]!=None and ca_list[i]!=None:
            if self.cb_list[i]==None:
                #print(1)
                ca_v=self.ca_list[i].get_vector().get_array()
                c_v=self.c_list[i].get_vector().get_array()
                n_v=self.n_list[i].get_vector().get_array()
                cb= self._calha1(n_v,c_v,ca_v) ###############
                #cb=PDB.vectors.Vector(cb)
                atom= cb
            else:
                atom = self.cb_list[i].coord

            vec1 = nb2- nb3
            vec2 = nb1- nb2
            vec3 = atom- nb1
            bond = mp.L_MO_ab(atom, nb1)
            angle = mp.get_angle(vec2, vec3)
            dhd = mp.get_dhd(vec1, vec2, vec3)
            inner_coord = bond, angle, dhd
            self.inner_cb.append(inner_coord)
        self.inner_coord_cb = torch.from_numpy(np.array(self.inner_cb,dtype= 'float32'))
        
class Get_Feature(object):
    def __init__(self, c, c2, device ,window = 16):
        self.device = device
        self.window = window
        self.c = c
        self.c2 = c2
        pass
    
    def get_feature(self):
        self._run()
    def get_ca_dist(self):
        return self.ca_dist_new
    
    def get_angle(self):
        return self.angle_d_m
    
    def get_index0(self):
        return self.index0 
    
    def get_num_cs(self):
        return self.num_cs
    
    def _run(self):
        coord_n = self.c[::3].clone()
        coord_ca = self.c[1::3].clone()
        coord_c = self.c[2::3].clone()
        coord_cb = self.c2
        A = coord_ca.clone()
        B = coord_ca.clone()
        aa_num = len(A)
        ca_dist = self._cal_ABdistance(A, B)
        self.num_cs = ca_dist.argsort()[:,1:self.window+1]        
        index0 = self._embed_index0()
        self.index0 = index0.to(self.device)
        self.ca_dist = ca_dist
        self.ca_dist_new = ca_dist[self.index0, self.num_cs]        
        angle_d_m = self._get_angle6_matrix(coord_ca, coord_cb, coord_n, coord_c)
        angle_d_m = list(angle_d_m)[0]
        self.angle_d_m = angle_d_m.view(-1, self.window, 6)
        
        
    def _embed_index0(self):
        h, w = self.num_cs.size()
        a = torch.arange(h).view(-1, 1)
        b = torch.ones(w).long().view(1, -1)
        ab = torch.mm(a, b)
        return ab
    
    def _cal_ABdistance(self,A,B):
        "calculate ca distance by matrix"
        B_t = torch.t(B)
        vecProd = torch.mm(A, B_t)
        SqA = A**2
        SqB = B**2
        #print("SqA:", SqA)
        #print("SqA:", SqB)
        sumSqA = torch.sum(SqA, 1)
        sumSqB = torch.sum(SqB, 1)
        a_len = len(sumSqA)
        b_len = len(sumSqB)
        sumSqA_w = sumSqA.view(1, -1)
        sumSqB_w = sumSqB.view(1, -1)
        sumSqBx = torch.t(sumSqB_w)
        eye = torch.ones((a_len,1),device = self.device)
        eye1 = torch.ones((1,a_len), device = self.device)
        sumSqAx = torch.mm(eye,sumSqA_w)
        sumSqBx_t = torch.mm(sumSqBx,eye1)
        SqED = sumSqAx + sumSqBx_t - 2 * vecProd
        SqED_0 =torch.where(SqED<0, torch.zeros(1,device = self.device), SqED)
        SqED_sq = torch.sqrt(SqED_0+1e-8)
        return SqED_sq

    def _get_angle6_matrix(self, coord_ca, coord_cb, coord_n, coord_c):
        """calculate angle by matrix

        """
        num_cs_1 = self.num_cs.contiguous().view(-1)
        h,w = self.num_cs.size()
        ca1 = coord_ca.unsqueeze(1)   #100*3
        ca2 = coord_ca[num_cs_1].view(-1,w,3) #100*16*3
        cb1 = coord_cb.unsqueeze(1)
        cb2 = coord_cb[num_cs_1].view(-1,w,3)
        n1 = coord_n.unsqueeze(1)
        n2 = coord_n[num_cs_1].view(-1,w,3)
        c1 = coord_c.unsqueeze(1)
        c2 = coord_c[num_cs_1].view(-1,w,3)
        eye = torch.ones(w, 1)
        #print(ca2.shape, ca1.shape )
        ca1_cb1 = (ca1 - cb1).repeat(1,w,1).view(-1,3) #100*3
        ca1_n1 = (ca1 - n1).repeat(1,w,1).view(-1,3)
        ca1_c1 = (ca1 - c1).repeat(1,w,1).view(-1,3)
        ca2_ca1 = (ca2 - ca1).view(-1,3)
        ca2_cb2 = (ca2 - cb2).view(-1,3)
        ca2_n2 = (ca2 - n2).view(-1,3)
        ca2_c2 = (ca2 - c2).view(-1,3)
        t1 = self._get_angle_matrix(ca1_cb1, ca2_ca1)
        t11 = self._get_angle_matrix(ca2_ca1, -ca2_cb2)
        t2 = self._get_angle_matrix(ca1_n1, ca2_ca1)
        t22 = self._get_angle_matrix(ca2_n2, -ca2_ca1)
        t3 = self._get_angle_matrix(ca1_c1, ca2_ca1)
        t33 = self._get_angle_matrix(ca2_c2, -ca2_ca1)
        angle = [t1, t2, t3, t11, t22, t33]
        angle_t = torch.t(torch.cat(angle).view(6, -1))
        yield angle_t
       
        
    def _get_angle_matrix(self, a, b):
        c = a * b  #16*3
        c_sum = torch.sum(c, 1)
        aa = a * a
        bb = b * b
        aa_norm = torch.sum(aa, 1)
        bb_norm = torch.sum(bb, 1)
        #print(bb_norm)
        ab = torch.rsqrt(aa_norm * bb_norm)
        tmp = c_sum * ab
        theta = torch.Tensor([np.pi]).to(self.device) - torch.acos(tmp)
        return theta


if __name__ == '__main__':    
    query_protein = '/state/partition1/cxy/CASP10-13/T0722/T0722TS035_1.pdb'
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    r = Read_pdb_feature()
    r.readpdb(query_protein)
    inner_coord_tensor = r.get_inner_coord()
    inner_coord_cb_tensor = r.get_inner_coord_cb()
    mainchain_coord_tensor = r.mc_tensor()
    cb_tensor = r.get_cb_tensor()

    net = NeRF_net(inner_coord_tensor)  #model
    net2 = NeRF_net_cb(inner_coord_cb_tensor)  
    c = net(mainchain_coord_tensor)
    c2 = net2(c)
    f = Get_Feature(c.to(device), c2.to(device), device)
    f.get_feature()
    dist = f.ca_dist_new
    angle =f.angle_d_m 
    num_cs = f.num_cs
    index0 = f.index0
    print("dist: %s, angle %s, num_cs: %s, index0: %s" %(dist.shape, angle.shape, num_cs.shape, index0.shape))
