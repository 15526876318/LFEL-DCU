import torch
import numpy as np
import math
import sys
import time
import math_p as mp
from random import choice
from Bio import PDB
import Bio.PDB as bio
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import calc_angle,calc_dihedral,vectors


       	
def rotation(r,v,theta):
    t1=r*np.cos(theta)
    t2=np.cross(v,r)
    t2=t2*np.sin(theta)
    vr=np.dot(v,r)
    t3=vr*v*(1-np.cos(theta))
    r=t1+t2+t3
    return r

def calha1(a,b,c):
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
    d=rotation(d,fabc,math.pi*108.0300/180.0)
    d=rotation(d,bc,math.pi*117.8600/180.0)
    d+=c
    return d

def get_id_chain_name(pdb_list_file):
    pdbid=[]
    pdbchain=[]
    with open(pdb_list_file, 'r') as f:
        lines = f.readlines()
    if not lines:
        print("read %s fail!" % pdb_list_file)
        sys.exit()
    for line in lines:
        line = line.strip('n')
        pdb = line.split()[0]
        pdbid.append(pdb[:4])
        pdbchain.append(pdb[4:])        
    return pdbid, pdbchain

def get_aa_list(res_list):
    aa_list = [a for a in res_list if PDB.is_aa(a)]
    return aa_list

def check_aa_id(aa_list):
    error = 0
    t=aa_list[0].get_id()[1]
    aa_list_full=[]
    for a in aa_list:
        while 1:
            if a.get_id()[1]<t:
                error=1
                break
            elif a.get_id()[1]==t:
                aa_list_full.append(a)
                t+=1
                break
            else:
                aa_list_full.append(None)
                t+=1
    if error==1:                 
        return 0
    return aa_list_full

def cal_depth(s, aa_list_full):
    depth=PDB.ResidueDepth(s)   
    dep_dict=depth.property_dict
    dps=[]
    for a in aa_list_full:
        try:
            aa_id=(a.get_parent().get_id(),a.get_id())
            if dep_dict.get(aa_id):
                dps.append(dep_dict[aa_id])
            else:
                dps.append([None,None])
        except:
            dps.append([None,None])
    dps=np.array(dps)
    return dps

def cal_hseab(s, aa_list_full):
    try:
        HSEA=PDB.HSExposureCA(s)
        HSEB=PDB.HSExposureCB(s)
    except:
        return 0,0
    HSEA_dict=HSEA.property_dict
    HSEB_dict=HSEB.property_dict
    hse_a=[]
    hse_b=[]
    for a in aa_list_full:
        try:
            aa_id=(a.get_parent().get_id(),a.get_id())
            if HSEA_dict.get(aa_id):
                hse_a.append(HSEA_dict[aa_id])
            else:
                hse_a.append([None,None,None])
        except:
            hse_a.append([None,None,None])
    hse_a=np.array(hse_a)
    for a in aa_list_full:
        try:
            aa_id=(a.get_parent().get_id(),a.get_id())
            if HSEB_dict.get(aa_id):
                hse_b.append(HSEB_dict[aa_id])
            else:
                hse_b.append([None,None,None])
        except:
            hse_b.append([None,None,None])

    hse_b=np.array(hse_b)
    return hse_a, hse_b

def get_seq(aa_list_full,t_dic):
    seq_list = ''
    for a in aa_list_full:
        try:
            t=a.get_resname()
            if t in t_dic:
                seq_list+=t_dic[t]
            else:
                seq_list+='X'
        except:
            seq_list+='X'
    return seq_list

def get_atom_list(aa_list_full, atom):
    atom_list=[]
    for a in aa_list_full:
        try:
            t=a[atom]
            atom_list.append(t)
        except:
            t=None
            atom_list.append(t)
    return atom_list
        
def cal_dist(ca_list):
    ca_num=len(ca_list)
    ca_dist=[]             #CA����
    for j in range(len(ca_list)):
        for k in range(len(ca_list)):
            if ca_list[j]!=None and ca_list[k]!=None:
                ca_dist.append(ca_list[j]-ca_list[k])
            else:
                ca_dist.append(None)    
    ca_dist=np.array(ca_dist)
    ca_dist=ca_dist.reshape(ca_num,ca_num)
    return ca_dist

def get_mask(ca_list):
    mask=[]    
    for j in range(len(ca_list)):
        if ca_list[j]!=None:
            mask.append(1)
        else:
            mask.append(0)
    return mask


def get_angle(a,b,device): 
    """
        calculate two tensor angle batch
        a(x,y,z), b(x,y,z)
    """  
    c = torch.dot(a,b)
    aa = torch.norm(a)
    bb = torch.norm(b)
    tmp = c/(aa*bb)
    if tmp > 1.0:
        tmp = torch.Tensor([1.0]).to(device)
    if tmp < -1.0:
        tmp = torch.Tensor([-1.0]).to(device)
    theta = torch.squeeze(torch.Tensor([np.pi]).to(device),0)-torch.acos(tmp)
    return theta

def get_angle5_ceshi(aa_num16, ca_list, cb_list, n_list, c_list, j,device):
    angle_t = []
    #cb1 = get_cb_vector(ca_list, cb_list, n_list, c_list, j)                
    for k in aa_num16:
        try:           
            ca1 = ca_list[j]
            cb1 = cb_list[j]
            ca2 = ca_list[k]
            vec1 = ca1-cb1
            vec2 = ca2-ca1
            t1 = get_angle(vec1, vec2,device)
            t1 = torch.unsqueeze(t1,0)
        except:
            print("cannot calculate angle! ")
            t1 = torch.Tensor([1])       
        try:
            ca1 = ca_list[j]
            ca2 = ca_list[k]
            cb2 = cb_list[k]
            vec1 = ca2-ca1
            vec2 = cb2-ca2
            t11 = get_angle(vec1, vec2,device)
            t11 = torch.unsqueeze(t11,0)
        except:
            t11 = torch.Tensor([1])
        try:            
            ca1 = ca_list[j]
            ca2 = ca_list[k]
            n_1 = n_list[j]
            vec1 = ca1-n_1
            vec2 = ca2-ca1
            t2 = get_angle(vec1, vec2,device)
            t2 = torch.unsqueeze(t2,0)
        except:
            t2 = torch.Tensor([1])
        try:
            ca1 = ca_list[j]
            ca2 = ca_list[k]
            n_2 = n_list[k]
            vec1 = ca2-n_2
            vec2 = ca1-ca2
            t22 = get_angle(vec1, vec2,device)
            t22 = torch.unsqueeze(t22,0)
        except:
            t22 = torch.Tensor([1])
               
        try:
            ca1 = ca_list[j]
            ca2 = ca_list[k]
            c_1 = c_list[j]
            vec1 = ca1-c_1
            vec2 = ca2-ca1
            t3 = get_angle(vec1, vec2,device)
            t3 = torch.unsqueeze(t3,0)
        except:
            t3 = torch.Tensor([1])

        try:
            ca1 = ca_list[j]
            ca2 = ca_list[k]
            c_2 = c_list[k]
            vec1 = ca2-c_2
            vec2 = ca1-ca2
            t33 = get_angle(vec1, vec2,device)
            t33 = torch.unsqueeze(t33,0)
        except:
            t33 = torch.Tensor([1])
                
        angle_t = [t1,t2,t3,t11,t22,t33]
        #print(angle_t)
        angle_t = torch.cat(angle_t)
        yield angle_t



def get_feature(c,c2,device):
    """
        calculate feature by for 
    """
    n_list = [c[i] for i in range(0, len(c),3)]
    ca_list = [c[i+1] for i in range(0, len(c),3)]
    c_list = [c[i+2] for i in range(0, len(c),3)]
    cb_list = c2
    ca_dist = []
    ca_num = len(ca_list)
    for j in range(ca_num):
        for k in range(ca_num):
            ca_ca = torch.norm(ca_list[j]-ca_list[k],2) 
            ca_ca = torch.unsqueeze(ca_ca,0)               
            ca_dist.append(ca_ca)
    ca_dist=torch.cat(ca_dist)
    ca_dist=ca_dist.view(ca_num,ca_num)
    ca_dist_cs=[]
    angle_cs=[]
    num_cs=[]
    #print("get_feature pre cost:%fs" % time_cost(start)) 
    for j in range(len(ca_dist)):
        t = ca_dist[j]
        s=t.argsort()
        aa_num16 = s[1:17]
        ca_dist_cs.append(t[s[1:17]])
        angle_d = get_angle5_ceshi(aa_num16, ca_list, cb_list, n_list, c_list, j,device)
        angle_d = list(angle_d)
        angle_cs.append(angle_d)
        num_cs.append(s[1:17])
    #print("get_feature gen cost:%fs" % time_cost(start)) 
    return ca_dist_cs, angle_cs, num_cs

def get_feature_cpu(c,c2, device,window=16):
    """
        calculate feature by cpu  
    """
    n_list = [c[i] for i in range(0, len(c),3)]
    ca_list = [c[i+1] for i in range(0, len(c),3)]
    c_list = [c[i+2] for i in range(0, len(c),3)]
    cb_list = c2
    ca_dist = []
    ca_num = len(ca_list)
    for j in range(ca_num):
        for k in range(ca_num):
            ca_ca = torch.norm(ca_list[j]-ca_list[k],2) 
            ca_ca = torch.unsqueeze(ca_ca,0)               
            ca_dist.append(ca_ca)
    ca_dist=torch.cat(ca_dist)
    ca_dist=ca_dist.view(ca_num,ca_num)
    ca_dist_cs=[]
    angle_cs=[]
    #num_cs=[]
    #print("get_feature pre cost:%fs" % time_cost(start)) 
    for j in range(len(ca_dist)):
        t = ca_dist[j]
        s=t.argsort()
        aa_num16 = s[1:17]
        ca_dist_cs.append(t[s[1:17]])
        angle_d = get_angle5_ceshi(aa_num16, ca_list, cb_list, n_list, c_list, j,device)
        angle_d = list(angle_d)
        angle_cs.append(angle_d)
        #num_cs.append(s[1:17])
    num_cs = get_cs_num(ca_dist,window)
    ab = ab_index(num_cs)
    ab = ab.to(device)
    #print("get_feature gen cost:%fs" % time_cost(start)) 
    return ca_dist_cs, angle_cs, num_cs, ca_dist, ab

def get_pdb_name(PATH, train_name):
    pdb_id = train_name[:4]
    chain = train_name[4]
    pdb_name = PATH + "pdb" + pdb_id.lower() + '.ent'
    return pdb_name, chain

def read_pdb(pdb_id, chain=None):
    p = PDBParser(PERMISSIVE=1)
    try:
        s = p.get_structure("1", pdb_id)
    except:
        print("*"*10)
        print(pdb_id)
        print("*"*10)
    #print("***%s***" % name)
    
    if chain:
        s = s[0][chain]
    else:
        s = s[0]
    #print(s)
    res_list = PDB.Selection.unfold_entities(s, 'R')  #read aminoacid
    aa_list = get_aa_list(res_list)
    #print(aa_list)
    aa_list_full = check_aa_id(aa_list)
    #print(aa_list_full)

    return aa_list_full 
    
def check_backbone_3(aa_list_full):
    """
        Checking the main chain for lack of atoms
    """
    ca_list = get_atom_list(aa_list_full, 'CA')
    c_list = get_atom_list(aa_list_full, 'C')
    n_list = get_atom_list(aa_list_full, 'N')
    if ca_list and c_list and n_list:
        return ca_list, c_list, n_list
    else:
        return 0, 0, 0   

def get_cb_list(aa_list_full):
    cb_list = []
    for a in aa_list_full:
        try:
            t = a['CB']
            cb_list.append(t)
        except:
            cb_list.append(None)
    return cb_list

def get_backbone(ca_list,c_list, n_list):
    """
        oputput: mainchain atom
        ca_list: Ca atom of all amino acids
        c_list: c atom of all amino acids
        n_list: n atom of all amino acids
    """
    mainchain = []
    for i in range(len(ca_list)):
        mainchain.append(n_list[i])
        mainchain.append(ca_list[i])
        mainchain.append(c_list[i])
    return mainchain



    
    
def get_inner_coord(mainchain):
    
    inner = []
    for i in range(3,len(mainchain)):
        atom = mainchain[i].coord
        #print(mainchain[i].id,atom)
        nb1 = mainchain[i-1].coord
        nb2 = mainchain[i-2].coord
        nb3 = mainchain[i-3].coord
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
        inner.append(inner_coord)
    a = torch.from_numpy(np.array(inner,dtype= 'float64'))
    return a

def get_inner_coord_cb(ca_list, cb_list, n_list, c_list):
    inner = []
    for i in range(len(cb_list)):
        nb3 = n_list[i].coord
        nb2 = ca_list[i].coord
        nb1 = c_list[i].coord
        #print(cb_list[i], c_list[i], n_list[i], ca_list[i])
        if cb_list[i]==None and c_list[i]!=None and n_list[i]!=None and ca_list[i]!=None:
            #print(1)
            ca_v=ca_list[i].get_vector().get_array()
            c_v=c_list[i].get_vector().get_array()
            n_v=n_list[i].get_vector().get_array()
            cb=calha1(n_v,c_v,ca_v)
            #cb=PDB.vectors.Vector(cb)
            atom= cb
        else:
            atom = cb_list[i].coord
        
        vec1 = nb2- nb3
        vec2 = nb1- nb2
        vec3 = atom- nb1
        #print(vec1,vec2,vec3)
        bond = mp.L_MO_ab(atom, nb1)
        #print(bond)
        angle = mp.get_angle(vec2, vec3)
        dhd = mp.get_dhd(vec1, vec2, vec3)
        inner_coord = bond, angle, dhd
        #print(inner_coord)
        inner.append(inner_coord)
    a = torch.from_numpy(np.array(inner,dtype= 'float64'))
    return a

def get_angle_matrix(a, b,device):
    c = a * b  #16*3
    #print(a,b)
    c_sum = torch.sum(c, 1)
    #print("c_sum", c_sum)
    #c_u = torch.squeeze(c_sum,1)
    #print(c_u.shape)
    aa = a * a
    bb = b * b
    aa_norm = torch.sum(aa, 1)
    bb_norm = torch.sum(bb, 1)
    #print(bb_norm)
    ab = torch.rsqrt(aa_norm * bb_norm)
    tmp = c_sum * ab
    theta = torch.Tensor([np.pi]).to(device) - torch.acos(tmp)
    return theta

def cal_ABdistance(A, B,device):
    "calculate ca distance by matrix"
    #print("A",A)
    #print("B",B)
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
    eye = torch.ones((a_len,1),device = device,dtype = torch.float64)
    eye1 = torch.ones((1,a_len), device = device,dtype = torch.float64)
    #print(eye.dtype)
    sumSqAx = torch.mm(eye,sumSqA_w)
    sumSqBx_t = torch.mm(sumSqBx,eye1)
    SqED = sumSqAx + sumSqBx_t - 2 * vecProd
    #print(SqED[:10])
    SqED_0 =torch.where(SqED<0, torch.zeros(1,device = device,dtype = torch.float64), SqED)
    SqED_sq = torch.sqrt(SqED_0+1e-8)
    return SqED_sq

def get_angle6_matrix(num_cs, coord_ca, coord_cb, coord_n, coord_c,device):
    """calculate angle by matrix
       
    """
    num_cs_1 = num_cs.contiguous().view(-1)
    h,w = num_cs.size()
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
    t1 = get_angle_matrix(ca1_cb1, ca2_ca1, device)
    t11 = get_angle_matrix(ca2_ca1, -ca2_cb2,device)
    t2 = get_angle_matrix(ca1_n1, ca2_ca1,device)
    t22 = get_angle_matrix(ca2_n2, -ca2_ca1,device)
    t3 = get_angle_matrix(ca1_c1, ca2_ca1,device)
    t33 = get_angle_matrix(ca2_c2, -ca2_ca1,device)
    angle = [t1, t2, t3, t11, t22, t33]
    angle_t = torch.t(torch.cat(angle).view(6, -1))
    yield angle_t

def get_cs_num(ca_dist,window=16):
    """ 
        output: aa num around target aa sorted by distance
        ca_dist: n*n tensor , n is sequence length
        window: cutoff of the round aa, defalt 16
    """
    return ca_dist.argsort()[:,1:window+1]

def ab_index(idx):
    h, w = idx.size()
    a = torch.arange(h).view(-1, 1)
    b = torch.ones(w).long().view(1, -1)
    ab = torch.mm(a, b)
    return ab

def get_feature_matrix(c, c2, device,window=16):
    coord_n = c[::3].clone()
    coord_ca = c[1::3].clone()
    coord_c = c[2::3].clone()
    coord_cb = c2
    A = coord_ca.clone()
    B = coord_ca.clone()
    aa_num = len(A)
    ca_dist = cal_ABdistance(A, B,device)
    num_cs = get_cs_num(ca_dist,window)
    ab = ab_index(num_cs)
    ab = ab.to(device)
    ca_dist_new = ca_dist[ab, num_cs] # distance 
    angle_d_m = get_angle6_matrix(num_cs, coord_ca, coord_cb, coord_n, coord_c,device)
    angle_d_m = list(angle_d_m)[0]
    angle_d_m = angle_d_m.view(-1, window, 6)
    return ca_dist_new, angle_d_m, num_cs, ca_dist, ab


def generate1(mask, num_cs, seq_list):
    ids=num_cs
    seq=seq_list    
    idx_nb = []
    idx_unb = []
    label = []
    index_h=[]
    index_nb =[]
    index_unb = []
    kk=0
    for i in range(len(mask)):
        nb_id = [j for j in range(-6+i,7+i) if j!=i]
        if mask[i]!=0:
            index_h.append(i)
            idx_nb_i = []
            idx_unb_i = []
            index_nb_i = []
            index_unb_i = []
            for j in range(len(ids[i])):
                if len(idx_unb_i)==10:
                    break
                if abs(ids[i][j]-i)>6:
                    index_unb_i.append(j)
                    if seq[ids[i][j]] in seqdic:
                        idx_unb_i.append(seqdic[seq[ids[i][j]]])
                    else:
                        idx_unb_i.append(20)
            while len(idx_unb_i) < 10:
                index_unb_i.append(-1)
                idx_unb_i.append(21)
            for a in nb_id:
                if a in ids[i]:
                    k=np.where(ids[i]==a)
                    k=int(k[0][0]) #duole yiwei suoyixuyao suoyin liangci
                    index_nb_i.append(k)
                    #for m1 in range(len(angle[i][k])):
                        #if angle[i][k][m1]==None:
                            #angle[i][k][m1]=random.random()*3.14
                    if seq[a] in seqdic:
                        idx_nb_i.append(seqdic[seq[a]])
                    else:
                        idx_nb_i.append(20)
                else:
                    index_nb_i.append(-1)
                    idx_nb_i.append(21)
   
            idx_unb.append(idx_unb_i)
            index_nb.append(index_nb_i)
            index_unb.append(index_unb_i)
            if seq[i] in seqdic:
                label.append(seqdic[seq[i]])
            else:
                label.append(20)
            kk+=1
    index_unb = torch.tensor(index_unb)
    index_nb1 = torch.tensor(index_nb)
    idx_nb = torch.tensor(idx_nb)
    idx_unb = torch.tensor(idx_unb)   
    index = torch.cat((index_unb,index_nb1), 1)
    idx = torch.cat((idx_unb, idx_nb),1)
    del index_nb1

    label1 = torch.tensor(label)
    index_h1 = torch.tensor(index_h)
    return  idx, label1, kk, index, index_h1

def opi_pro(train_name, PDB_PATH,FG_N):
    #train_name = TRAIN[1]    
    print(train_name,DEVICE)
    #train_name = TRAIN[num[1]]
    pdb_name, chain = get_pdb_name(PDB_PATH, train_name)
    aa_list_full = read_pdb(pdb_name, chain)
    ca_list, c_list, n_list= check_backbone_3(aa_list_full)
    cb_list = get_cb_list(aa_list_full)
    if not ca_list or not c_list or not n_list:
        print("coord error!")
        sys.exit()

    backbone  = get_backbone(ca_list, c_list, n_list)
    mask = get_mask(ca_list)
    seq_list = get_seq(aa_list_full,t_dic)

    inner_tensor = get_inner_coord(backbone)
    inner_coord_cb_tensor = get_inner_coord_cb(ca_list, cb_list, n_list, c_list)
    mainchain_coord_tensor = torch.from_numpy(np.array([i.coord for i in backbone]))

    ### net, lossfunction, optimizer###    
    net = NeRF_net(inner_tensor)
    net2 = NeRF_net_cb(inner_coord_cb_tensor) 
    net_pnerf = PNERF_net(inner_tensor)
    
    criterion = nn.NLLLoss()
    if FG_N:
        print("PNERF...")
        optimizer = optim.SGD([net_pnerf.psi,net_pnerf.phy], lr=LR)
    else:
        optimizer = optim.SGD(net.dhd_v, lr=LR)   
    
    ###refinement###
    for i in range(6):
        optimizer.zero_grad()
        if FG_N:	
            c =net_pnerf(FG_N, mainchain_coord_tensor)
        else:
            c = net(mainchain_coord_tensor)
        c2 = net2(c)
        c = c.to(DEVICE)
        c2 = c2.to(DEVICE)
        dist, angle, num_cs,cad,abb = get_feature_matrix(c, c2,DEVICE,WINDOW)	
        num_cs1 = num_cs.cpu().numpy()
        idx_t, y, kkn, index_t, index_h = generate1(mask, num_cs1, seq_list) #
        idx_t=idx_t.to(DEVICE)
        index_t = index_t.to(DEVICE)
        index_h = index_h.to(DEVICE)
        y = y.to(DEVICE)
        x = g_data_net(dist, angle, idx_t, index_t, index_h,DEVICE)
        #dist, angle, num_cs = get_feature(c,c2)
        #x, idx_, dis_, angle_t, y, kkn = g_data_net_cpu1(mask, num_cs , dist, angle, seq_list,seqdic)
        #x=x.to(DEVICE)
        #dis_=dis_.to(DEVICE)
        #idx_=idx_.to(DEVICE)
        #angle_t=angle_t.to(DEVICE)
        outputs=model(x)
        loss = criterion(F.log_softmax(outputs,dim=1), y)
        loss.backward()
        
        optimizer.step()
        time1 = time_cost(START)
        print(" iter:%d ,opi_loss %f time_cost%8.4fs\n" % (i, loss.item(), time1))
        
def c2inner_coord(backbone):
    backbone_dis = [(backbone[i]-backbone[i-1]) for i in range(3,len(backbone))]
    backbone_angle = [ np.pi-calc_angle(backbone[i-2].get_vector(), backbone[i-1].get_vector(), backbone[i].get_vector()) for i in range(3,len(backbone))]
    backbone_dihedral = [ calc_dihedral(backbone[i].get_vector(), backbone[i-1].get_vector(), backbone[i-2].get_vector(),backbone[i-3].get_vector()) for i in range(3,len(backbone))]
    inner_tensor2 = np.array([[backbone_dis[i] ,backbone_angle[i],backbone_dihedral[i]] for i in range(len(backbone_dis))],dtype = 'float64')
    inner_tensor2 = torch.from_numpy(inner_tensor2)
    return inner_tensor2
