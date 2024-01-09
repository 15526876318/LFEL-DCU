import torch
import numpy as np
import random
import math
from random import choice
from Bio import PDB
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.DSSP import DSSP
#import pdb_read.SCE10 as sc
#import math_p as mp
#np.random.seed(1)
#random.seed(1)

seqdic={'A':0, 'R':1, 'D':2, 'C':3, 'Q':4, 'E':5, 'H':6, 'I':7, 'G':8, 'N':9, 'L':10, 
        'K':11, 'M':12, 'F':13, 'P':14, 'S':15, 'T':16, 'W':17, 'Y':18, 'V':19}
t_dic={'ALA':'A','VAL':'V','LEU':'L','ILE':'I','PHE':'F','TRP':'W','MET':'M','PRO':'P',\
       'GLY':'G','SER':'S','THR':'T','CYS':'C','TYR':'Y','ASN':'N','GLN':'Q','HIS':'H',\
       'LYS':'K','ARG':'R','ASP':'D','GLU':'E'}
ss_dict_823 = {'H':'H', 'G':'H', 'I':'H', 'E':'E','B':'C', 'T':'C', 'S':'C','-':'C'}

pssm_train_data = '/home/cxy/旧电脑/PycharmProjects/contact_map/cullpdb_train.npy'
pssm_test_data = '/home/cxy/旧电脑/PycharmProjects/contact_map/cullpdb_train.npy'
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device(cuda if torch.cuda.is_available() else "cpu")
window = 16

def rotation(r,v,theta):
    t1=r*np.cos(theta)
    t2=np.cross(v,r)
    t2=t2*np.sin(theta)
    vr=np.dot(v,r)
    t3=vr*v*(1-np.cos(theta))
    r=t1+t2+t3
    return r

def calha1(a,b,c):
    " 计算gly侧链H坐标"
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


def get_dssp(query_protein,chain):
    ss = ''
    p = PDBParser(PERMISSIVE=1)
    s = p.get_structure("1",query_protein)
    s = s[0]
    dssp = DSSP(s,query_protein)
    for i,info in enumerate(dssp.property_keys):
        if info[0] ==chain:
            ss= ss + ss_dict_823[dssp.property_list[i][2]]
    return ss 

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
    #计算氨基酸到蛋白质表面距离
    try:
        depth=PDB.ResidueDepth(s)   
        dep_dict=depth.property_dict
    except:
        return 0
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

def get_seq(aa_list_full):
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

def get_atom_list_npy(aa_list_full, atom):
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
    ca_dist=[]             #CA距离
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
    mask=[]    #是否有CA
    for j in range(len(ca_list)):
        if ca_list[j]!=None:
            mask.append(1)
        else:
            mask.append(0)
    return mask

def time_cost(start):
    time_cost = time.time() - start
    print(time_cost)

def get_cb_vector(ca_list, cb_list, n_list, c_list, j):
    if cb_list[j] != None:
        cb = cb_list[j].get_vector()
    elif cb_list[j] == None and c_list[j]!=None and n_list[j]!=None and ca_list[j]!=None:
        ca_v=ca_list[j].get_vector().get_array()
        c_v=c_list[j].get_vector().get_array()
        n_v=n_list[j].get_vector().get_array()
        cb=calha1(n_v,c_v,ca_v)
        cb=PDB.vectors.Vector(cb)
    else:
        cb = cb_list[j]
    return cb

def get_pssm(data1,data2):
    pssm_dict = {}
    data=np.load(data1,allow_pickle=True).item()
    test_data=np.load(data2,allow_pickle=True).item()
    train_name = data['name']
    train_pssm = data['pssm']
    test_name = data['name']
    test_pssm = data['pssm']
    train_pssm_n = (train_pssm-np.mean(train_pssm))/np.std(train_pssm)
    test_pssm_n = (test_pssm-np.mean(test_pssm))/np.std(test_pssm)    
    for i,name in enumerate(pdb_name):
        pssm_dict[name] = pdb_pssm[i]
    for i,name in enumerate(test_data):
        pssm_dict[name] = pdb_pssm[i]
    return pssm_dict


def get_angle5_ceshi(aa_num16,ca_list, cb_list, n_list, c_list, j):
    angle_t = []
    cb1 = get_cb_vector(ca_list, cb_list, n_list, c_list, j)                
    for k in aa_num16:
        try:
            ca1=ca_list[j].get_vector()
            ca2=ca_list[k].get_vector()            
            t1=PDB.vectors.calc_angle(cb1,ca1,ca2)
        except:
            t1 = None            
        try:
            ca1=ca_list[j].get_vector()
            ca2=ca_list[k].get_vector()
            cb2 = get_cb_vector(ca_list, cb_list, n_list, c_list, k)
            t11 = PDB.vectors.calc_angle(cb2,ca2,ca1)
        except:
            t11 = None
            
        try:
            ca1=ca_list[j].get_vector()
            ca2=ca_list[k].get_vector()
            n_1 = n_list[j].get_vector()
            t2=PDB.vectors.calc_angle(n_1,ca1,ca2)
        except:
            t2 = None
        try:
            ca1 = ca_list[j].get_vector()
            ca2 = ca_list[k].get_vector()
            n_2 = n_list[k].get_vector()
            t22 = PDB.vectors.calc_angle(n_2,ca2,ca1)
        except:
            t22 = None        
        
        try:    
            ca1=ca_list[j].get_vector()
            ca2=ca_list[k].get_vector()
            c_1 = c_list[j].get_vector()
            t3 = PDB.vectors.calc_angle(c_1,ca1,ca2)
        except:
            t3 = None
        try:    
            ca1=ca_list[j].get_vector()
            ca2=ca_list[k].get_vector()
            c_2 = c_list[k].get_vector()
            t33 = PDB.vectors.calc_angle(c_2,ca2,ca1)
        except:
            t33 = None            
            
        angle_t=t1,t2,t3,t11,t22,t33        
        yield angle_t

def get_cs_num(ca_dist,window):
    " 得到的周围氨基酸排序"
    return ca_dist.argsort()[:,1:window+1]

def get_backbone(ca_list,c_list, n_list):
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
    a = torch.from_numpy(np.array(inner,dtype= 'float32'))
    return a

def get_inner_coord_cb(ca_list, cb_list, n_list, c_list):
    "cal cb inner coord"
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
    a = torch.from_numpy(np.array(inner,dtype= 'float32'))
    return a

def get_angle(a,b):   
    c = torch.dot(a,b)
    aa = torch.norm(a)
    bb = torch.norm(b)
    tmp = c/(aa*bb)
    if tmp > 1.0:
        tmp = 1.0
    if tmp < -1.0:
        tmp = -1.0
    theta = torch.squeeze(torch.Tensor([np.pi]),0)-torch.acos(tmp)
    return theta#

def get_angle_matrix(a, b):
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

def get_angle5_ceshi_torch(aa_num16, ca_list, cb_list, n_list, c_list, j):
    "torch banben"
    angle_t = []
    #cb1 = get_cb_vector(ca_list, cb_list, n_list, c_list, j)                
    for k in aa_num16:
        try:           
            ca1 = ca_list[j]
            cb1 = cb_list[j]
            ca2 = ca_list[k]
            vec1 = ca1-cb1
            vec2 = ca2-ca1
            #print(vec1)
            t1 = get_angle(vec1, vec2)
            t1 = torch.unsqueeze(t1,0)
        except:
            t1 = torch.Tensor([1])       
        try:
            ca1 = ca_list[j]
            ca2 = ca_list[k]
            cb2 = cb_list[k]
            vec1 = ca2-ca1
            vec2 = cb2-ca2
            t11 = get_angle(vec1, vec2)
            t11 = torch.unsqueeze(t11,0)
        except:
            t11 = torch.Tensor([1])
        try:            
            ca1 = ca_list[j]
            ca2 = ca_list[k]
            n_1 = n_list[j]
            vec1 = ca1-n_1
            vec2 = ca2-ca1
            t2 = get_angle(vec1, vec2)
            t2 = torch.unsqueeze(t2,0)
        except:
            t2 = torch.Tensor([1])
        try:
            ca1 = ca_list[j]
            ca2 = ca_list[k]
            n_2 = n_list[k]
            vec1 = ca2-n_2
            vec2 = ca1-ca2
            t22 = get_angle(vec1, vec2)
            t22 = torch.unsqueeze(t22,0)
        except:
            t22 = torch.Tensor([1])
               
        try:
            ca1 = ca_list[j]
            ca2 = ca_list[k]
            c_1 = c_list[j]
            vec1 = ca1-c_1
            vec2 = ca2-ca1
            t3 = get_angle(vec1, vec2)
            t3 = torch.unsqueeze(t3,0)
        except:
            t3 = torch.Tensor([1])

        try:
            ca1 = ca_list[j]
            ca2 = ca_list[k]
            c_2 = c_list[k]
            vec1 = ca2-c_2
            vec2 = ca1-ca2
            t33 = get_angle(vec1, vec2)
            t33 = torch.unsqueeze(t33,0)
        except:
            t33 = torch.Tensor([1])
                
        angle_t = [t1,t2,t3,t11,t22,t33]
        #print(angle_t)
        angle_t = torch.cat(angle_t)
        yield angle_t

def get_feature(c,c2):
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
        angle_d = get_angle5_ceshi_torch(aa_num16, ca_list, cb_list, n_list, c_list, j)
        angle_d = list(angle_d)
        angle_cs.append(angle_d)
        num_cs.append(s[1:17])
    #print("get_feature gen cost:%fs" % time_cost(start)) 
    return ca_dist_cs, angle_cs, num_cs

def cal_ABdistance(A, B):
    "cal AB distance by matrix"
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
    eye = torch.ones((a_len,1),device = device)
    eye1 = torch.ones((1,a_len), device = device)
    sumSqAx = torch.mm(eye,sumSqA_w)
    sumSqBx_t = torch.mm(sumSqBx,eye1)
    SqED = sumSqAx + sumSqBx_t - 2 * vecProd
    #print(SqED[:10])
    SqED_0 =torch.where(SqED<0, torch.zeros(1,device = device), SqED)
    SqED_sq = torch.sqrt(SqED_0+1e-8)
    return SqED_sq

def get_angle6_matrix(num_cs, coord_ca, coord_cb, coord_n, coord_c):
    "cal 6 angle by matrix"
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
    t1 = get_angle_matrix(ca1_cb1, ca2_ca1)
    #print(t1[:32])
    t11 = get_angle_matrix(ca2_ca1, -ca2_cb2)
    t2 = get_angle_matrix(ca1_n1, ca2_ca1)
    t22 = get_angle_matrix(ca2_n2, -ca2_ca1)
    t3 = get_angle_matrix(ca1_c1, ca2_ca1)
    t33 = get_angle_matrix(ca2_c2, -ca2_ca1)
    angle = [t1, t2, t3, t11, t22, t33]
    angle_t = torch.t(torch.cat(angle).view(6, -1))
    yield angle_t

def get_feature_matrix(c, c2, window):
    "Extracting features by matrix"
    coord_n = c[::3].clone()
    coord_ca = c[1::3].clone()
    coord_c = c[2::3].clone()
    coord_cb = c2
    A = coord_ca.clone()
    B = coord_ca.clone()
    aa_num = len(A)
    ca_dist = cal_ABdistance(A, B)
    num_cs = get_cs_num(ca_dist,window)
    ab = ab_index(num_cs)
    ab = ab.to(device)
    ca_dist_new = ca_dist[ab, num_cs]
    angle_d_m = get_angle6_matrix(num_cs, coord_ca, coord_cb, coord_n, coord_c)
    angle_d_m = list(angle_d_m)[0]
    angle_d_m = angle_d_m.view(-1, window, 6)
    #return ca_dist_new, angle_d_m, num_cs
    return ca_dist_new, angle_d_m, num_cs, ca_dist, ab


def get_pdb_name(PATH, train_name):
    pdb_id = train_name[:4]
    chain = train_name[4]
    pdb_name = PATH + "pdb" + pdb_id.lower() + '.ent'
    return pdb_name, chain

def read_pdb(pdb_id, chain):
    p = PDBParser(PERMISSIVE=0)
    s = p.get_structure("1", pdb_id)
    #print("***%s***" % name)
    s = s[0][chain]
    res_list = PDB.Selection.unfold_entities(s, 'R')  #read aminoacid
    aa_list = get_aa_list(res_list)
    aa_list_full = check_aa_id(aa_list)
    return aa_list_full

def cal_ca_dist(ca_list):
    ca_num = len(ca_list)
    ca_dist = np.array([
        ca_list[i] - ca_list[j] for i in range(ca_num) for j in range(ca_num)
    ])
    ca_dist = ca_dist.reshape(-1, ca_num)
    return ca_dist


def get_atom_list(aa_list_full, atom):
    atom_list = []
    for i, a in enumerate(aa_list_full):
        try:
            t = a[atom]
            atom_list.append(t)
        except:
            #print("res %d not have atom %s" % (i,atom))
            #sys.exit()
            return 0
    return atom_list


def get_cb_list(aa_list_full):
    cb_list = []
    for a in aa_list_full:
        try:
            t = a['CB']
            cb_list.append(t)
        except:
            cb_list.append(None)
    return cb_list


def check_backbone_3(aa_list_full):
    ca_list = get_atom_list(aa_list_full, 'CA')
    c_list = get_atom_list(aa_list_full, 'C')
    n_list = get_atom_list(aa_list_full, 'N')
    if ca_list and c_list and n_list:
        return ca_list, c_list, n_list
    else:
        return 0, 0, 0


"mutation_dic_set ,get_first_mutation_site, get_mu_num generate mutation list"
def mutation_dic_set(ca_dist, DIS_CUTOFF):
    ca_dist = np.round(ca_dist,2)
    dic_muset = {}
    for i in range(len(ca_dist)):
        non_num =np.where(ca_dist[i]>DIS_CUTOFF)
        #print(non_num[0])
        dic_muset[i] = non_num[0]
    return dic_muset

def get_first_mutation_site(dic_muset,ca_dist):
    mu1 = np.random.randint(0,len(ca_dist))
    set1 = set(dic_muset[mu1])
    return mu1, set1

def get_mu_num(dic_muset, mu1, set1, mutation_list):
    mutation_list.append(mu1)
    if not list(set1):
        return mutation_list
    else:        
        mu2 = choice(list(set1))
        set2 = dic_muset[mu2]
        set12 = set(set1.intersection(set2))
        return get_mu_num(dic_muset, mu2, set12,mutation_list)

def ab_index(idx):
    h, w = idx.size()
    a = torch.arange(h).view(-1, 1)
    b = torch.ones(w).long().view(1, -1)
    ab = torch.mm(a, b)
    return ab

def OP_angle_dhd(res_name, group_name_list1, group_coord_list1):
    op_Axis = {}
    for i ,group in enumerate(group_name_list1):
        if group == 'ca_xyz':
            A = group_coord_list1[i]
            A1 = group_coord_list1[1] #n
            A2 = group_coord_list1[2] #co 
            x,y,z = Orientation_angle1(A,A1,A2)
            op_Axis[group] = x,y,z,A
        if group == 'n_xyz':
            A = group_coord_list1[i]
            A1 = group_coord_list1[0] #ca
            A2 = group_coord_list1[2] #co
            x,y,z = Orientation_angle2(A,A1,A2)
            op_Axis[group] = x,y,z,A
        if group == 'co_xyz':
            A = group_coord_list1[i]
            A1 = group_coord_list1[0] #ca
            A2 = group_coord_list1[1] #n
            x,y,z = Orientation_angle2(A,A1,A2) 
            op_Axis[group] = x,y,z,A
        if res_name == 'ALA':
            if group == 'ch3_xyz':
                A = group_coord_list1[i]
                A1 = group_coord_list1[0] #ca
                A2 = group_coord_list1[1] #n
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
        if res_name == 'VAL':
            if group == 'ch_xyz':
                A = group_coord_list1[i]
                A1 = group_coord_list1[0] #ca
                A2 = group_coord_list1[1] #n 
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'ch31_xyz':
                A = group_coord_list1[i]
                A1 = group_coord_list1[3] #ca
                A2 = group_coord_list1[0] #n 
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'ch32_xyz':
                A = group_coord_list1[i]
                A1 = group_coord_list1[3] #ca
                A2 = group_coord_list1[0] #n
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
        if res_name == 'LEU':
            if group == 'ch2_xyz':
                A = group_coord_list1[i]
                A1 = group_coord_list1[0] #ca
                A2 = group_coord_list1[1] #n 
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'ch_xyz':
                A = group_coord_list1[i]
                A1 = group_coord_list1[3] #ca
                A2 = group_coord_list1[0] #n   
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'ch31_xyz':
                A = group_coord_list1[i]
                A1 = group_coord_list1[4] #ca
                A2 = group_coord_list1[3] #n 
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'ch32_xyz':
                A = group_coord_list1[i]
                A1 = group_coord_list1[4] #ca
                A2 = group_coord_list1[3] #n 
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
        if res_name == 'PHE':
            if group == 'ch_xyz':
                A = group_coord_list1[i]
                A1 = group_coord_list1[0] #ca
                A2 = group_coord_list1[1] #n 
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'ben_xyz':
                A = group_coord_list1[i]
                A1 = group_coord_list1[3] #ca
                A2 = group_coord_list1[0] #n 
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
        if res_name =='PRO':
            if group == 'ch21_xyz':
                A = group_coord_list1[i]
                A1 = group_coord_list1[0] #ca
                A2 = group_coord_list1[1] #n 
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'ch22_xyz':
                A = group_coord_list1[i]
                A1 = group_coord_list1[3] #ca
                A2 = group_coord_list1[0] #n 
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'ch23_xyz':
                A = group_coord_list1[i]
                A1 = group_coord_list1[4] #ca
                A2 = group_coord_list1[3] #n
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
        if res_name == 'MET':
            if group == 'ch21_xyz': #ch21_xyz', 'ch22_xyz', 's_xyz', 'ch3_xyz'
                A = group_coord_list1[i]
                A1 = group_coord_list1[0] #ca
                A2 = group_coord_list1[1] #n
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'ch22_xyz':
                A = group_coord_list1[i]
                A1 = group_coord_list1[3] #ca
                A2 = group_coord_list1[0] #n
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 's_xyz':
                A = group_coord_list1[i]
                A1 = group_coord_list1[4] #ca
                A2 = group_coord_list1[3] #n
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'ch3_xyz':
                A = group_coord_list1[i]
                A1 = group_coord_list1[5] #ca
                A2 = group_coord_list1[4] #n
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
        if res_name == 'TRP':
            if group == 'ch1_xyz': #ch21_xyz', 'ch22_xyz', 's_xyz', 'ch3_xyz'
                A = group_coord_list1[i]
                A1 = group_coord_list1[0] #ca
                A2 = group_coord_list1[1] #n
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'huan2_xyz':
                A = group_coord_list1[i]
                A1 = group_coord_list1[3] #ca
                A2 = group_coord_list1[0] #n 
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
        if res_name == 'CYS':
            if group == 'ch_xyz': #ch21_xyz', 'ch22_xyz', 's_xyz', 'ch3_xyz'
                A = group_coord_list1[i]
                A1 = group_coord_list1[0] #ca
                A2 = group_coord_list1[1] #n  
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'sh_xyz':
                A = group_coord_list1[i]
                A1 = group_coord_list1[3] #ca
                A2 = group_coord_list1[0] #n
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
        if res_name == 'SER':
            if group == 'ch2_xyz': #ch21_xyz', 'ch22_xyz', 's_xyz', 'ch3_xyz'
                A = group_coord_list1[i]
                A1 = group_coord_list1[0] #ca
                A2 = group_coord_list1[1] #n
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'oh_xyz':
                A = group_coord_list1[i]
                A1 = group_coord_list1[3] #ca
                A2 = group_coord_list1[0] #n
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
        if res_name == 'THR':
            if group == 'ch2_xyz': #ch21_xyz', 'ch22_xyz', 's_xyz', 'ch3_xyz'
                A = group_coord_list1[i]
                A1 = group_coord_list1[0] #ca
                A2 = group_coord_list1[1] #n
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'oh_xyz':
                A = group_coord_list1[i]
                A1 = group_coord_list1[3] #ca
                A2 = group_coord_list1[0] #n
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'ch3_xyz':
                A = group_coord_list1[i]
                A1 = group_coord_list1[3] #ca
                A2 = group_coord_list1[0] #n
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
        if res_name == 'ASN':
            if group == 'ch2_xyz': #ch21_xyz', 'ch22_xyz', 's_xyz', 'ch3_xyz'
                A = group_coord_list1[i]
                A1 = group_coord_list1[0] #ca
                A2 = group_coord_list1[1] #n
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'co2_xyz': #ch21_xyz', 'ch22_xyz', 's_xyz', 'ch3_xyz'
                A = group_coord_list1[i]
                A1 = group_coord_list1[3] #ca
                A2 = group_coord_list1[0] #n
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'nh2_xyz':
                A = group_coord_list1[i]
                A1 = group_coord_list1[4] #ca
                A2 = group_coord_list1[3] #n  
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
        if res_name == 'GLN':
            if group == 'ch21_xyz': #ch21_xyz', 'ch22_xyz', 's_xyz', 'ch3_xyz'
                A = group_coord_list1[i]
                A1 = group_coord_list1[0] #ca
                A2 = group_coord_list1[1] #n
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'ch22_xyz': #ch21_xyz', 'ch22_xyz', 's_xyz', 'ch3_xyz'
                A = group_coord_list1[i]
                A1 = group_coord_list1[3] #ca
                A2 = group_coord_list1[0] #n
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'co2_xyz': #ch21_xyz', 'ch22_xyz', 's_xyz', 'ch3_xyz'
                A = group_coord_list1[i]
                A1 = group_coord_list1[4] #ca
                A2 = group_coord_list1[3] #n
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'nh2_xyz': #ch21_xyz', 'ch22_xyz', 's_xyz', 'ch3_xyz'
                A = group_coord_list1[i]
                A1 = group_coord_list1[5] #ca
                A2 = group_coord_list1[4] #n
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
        if res_name == 'ARG':
            if group == 'ch21_xyz': #
                A = group_coord_list1[i]
                A1 = group_coord_list1[0] #ca
                A2 = group_coord_list1[1] #n
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'ch22_xyz': #ch21_xyz', 'ch22_xyz', 's_xyz', 'ch3_xyz'
                A = group_coord_list1[i]
                A1 = group_coord_list1[3] #ca
                A2 = group_coord_list1[0] #n
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'ch23_xyz': #ch21_xyz', 'ch22_xyz', 's_xyz', 'ch3_xyz'
                A = group_coord_list1[i]
                A1 = group_coord_list1[4] #ca
                A2 = group_coord_list1[3] #n
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'nh_xyz': #ch21_xyz', 'ch22_xyz', 's_xyz', 'ch3_xyz'
                A = group_coord_list1[i]
                A1 = group_coord_list1[5] #ca
                A2 = group_coord_list1[4] #n
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'c_xyz': #ch21_xyz', 'ch22_xyz', 's_xyz', 'ch3_xyz'
                A = group_coord_list1[i]
                A1 = group_coord_list1[6] #ca
                A2 = group_coord_list1[5] #n
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'nh21_xyz': #ch21_xyz', 'ch22_xyz', 's_xyz', 'ch3_xyz'
                A = group_coord_list1[i]
                A1 = group_coord_list1[7] #ca
                A2 = group_coord_list1[6] #n
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'nh22_xyz': #ch21_xyz', 'ch22_xyz', 's_xyz', 'ch3_xyz'
                A = group_coord_list1[i]
                A1 = group_coord_list1[7] #ca
                A2 = group_coord_list1[6] #n
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
        if res_name == 'TYR':
            if group == 'ch2_xyz': #ch21_xyz', 'ch22_xyz', 's_xyz', 'ch3_xyz'
                A = group_coord_list1[i]
                A1 = group_coord_list1[0] #ca
                A2 = group_coord_list1[1] #n 
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'ben_oh_xyz':
                A = group_coord_list1[i]
                A1 = group_coord_list1[3] #ca
                A2 = group_coord_list1[0] #n 
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
        if res_name == 'HIS':
            if group == 'ch2_xyz': #ch21_xyz', 'ch22_xyz', 's_xyz', 'ch3_xyz'
                A = group_coord_list1[i]
                A1 = group_coord_list1[0] #ca
                A2 = group_coord_list1[1] #n  
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'huan3_xyz':
                A = group_coord_list1[i]
                A1 = group_coord_list1[3] #ca
                A2 = group_coord_list1[0] #n
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
        if res_name == 'ASP':
            if group == 'ch2_xyz': #ch21_xyz', 'ch22_xyz', 's_xyz', 'ch3_xyz'
                A = group_coord_list1[i]
                A1 = group_coord_list1[0] #ca
                A2 = group_coord_list1[1] #n
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'cooh_xyz':
                A = group_coord_list1[i]
                A1 = group_coord_list1[3] #ca
                A2 = group_coord_list1[0] #n
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
        if res_name == 'GLU':
            if group == 'ch21_xyz': #ch21_xyz', 'ch22_xyz', 's_xyz', 'ch3_xyz'
                A = group_coord_list1[i]
                A1 = group_coord_list1[0] #ca
                A2 = group_coord_list1[1] #n  
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'ch22_xyz': #ch21_xyz', 'ch22_xyz', 's_xyz', 'ch3_xyz'
                A = group_coord_list1[i]
                A1 = group_coord_list1[3] #ca
                A2 = group_coord_list1[0] #n 
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'cooh_xyz':
                A = group_coord_list1[i]
                A1 = group_coord_list1[4] #ca
                A2 = group_coord_list1[3] #n
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
        if res_name == 'LYS':
            if group == 'ch21_xyz': #ch21_xyz', 'ch22_xyz', 's_xyz', 'ch3_xyz'
                A = group_coord_list1[i]
                A1 = group_coord_list1[0] #ca
                A2 = group_coord_list1[1] #n  
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'ch22_xyz': #ch21_xyz', 'ch22_xyz', 's_xyz', 'ch3_xyz'
                A = group_coord_list1[i]
                A1 = group_coord_list1[3] #ca
                A2 = group_coord_list1[0] #n 
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'ch23_xyz':
                A = group_coord_list1[i]
                A1 = group_coord_list1[4] #ca
                A2 = group_coord_list1[3] #n 
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'ch24_xyz':
                A = group_coord_list1[i]
                A1 = group_coord_list1[5] #ca
                A2 = group_coord_list1[4] #n  
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'nh3_xyz':
                A = group_coord_list1[i]
                A1 = group_coord_list1[6] #ca
                A2 = group_coord_list1[5] #n  
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
        if res_name == 'ILE':
            if group == 'ch_xyz': #ch21_xyz', 'ch22_xyz', 's_xyz', 'ch3_xyz'
                A = group_coord_list1[i]
                A1 = group_coord_list1[0] #ca
                A2 = group_coord_list1[1] #n   
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'ch31_xyz': #ch21_xyz', 'ch22_xyz', 's_xyz', 'ch3_xyz'
                A = group_coord_list1[i]
                A1 = group_coord_list1[3] #ca
                A2 = group_coord_list1[0] #n 
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'ch32_xyz':
                A = group_coord_list1[i]
                A1 = group_coord_list1[3] #ca
                A2 = group_coord_list1[0] #n 
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
            if group == 'ch2_xyz':
                A = group_coord_list1[i]
                A1 = group_coord_list1[4] #ca
                A2 = group_coord_list1[3] #n 
                x,y,z = Orientation_angle2(A,A1,A2)
                op_Axis[group] = x,y,z,A
    return op_Axis
