import numpy as np
#from sklearn.cluster import OPTICS
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import time
import argparse
import sys
import os 
from module import read_pdb,check_backbone_3,get_cb_list,get_backbone,get_mask,get_seq,get_inner_coord,\
        get_inner_coord_cb, get_feature, get_feature_matrix,c2inner_coord
from nn import NeRF_net_cb, NeRF_net, g_data_net_cpu,Sim_p,g_data_net_tian0, PNERF_net
from torch.multiprocessing import Process
import Bio.SVDSuperimposer as SVD
from tqdm import tqdm


def time_cost(start):
    time_cost = time.time() - start
    time_cost = round(time_cost, 4)
    return time_cost

def get_pdb_name(PATH, train_name):
    pdb_id = train_name[:4]
    chain = train_name[4]
    #print(train_name)
    pdb_name = PATH + "pdb" + pdb_id.lower() + '.ent'
    return pdb_name, chain

t_dic={'ALA':'A','VAL':'V','LEU':'L','ILE':'I','PHE':'F','TRP':'W','MET':'M','PRO':'P',\
       'GLY':'G','SER':'S','THR':'T','CYS':'C','TYR':'Y','ASN':'N','GLN':'Q','HIS':'H',\
       'LYS':'K','ARG':'R','ASP':'D','GLU':'E'}
seqdic={'A':0, 'R':1, 'D':2, 'C':3, 'Q':4, 'E':5, 'H':6, 'I':7, 'G':8, 'N':9, 'L':10, \
    'K':11, 'M':12, 'F':13, 'P':14, 'S':15, 'T':16, 'W':17, 'Y':18, 'V':19, 'X':20}

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
            idx_nb.append(idx_nb_i)
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


def save_opi_strcture(c,c2,aa_list_full,j, path):
    res_num= 0
    filename = (path+"/%s.pdb" % (j+1))
    #print("refinemnt pdb file:",filename)
    #filename = ("result/%s.pdb" % (j+1))
    serial_num = 0
    chain = 'A'
    with open (filename,'w') as outfile:    
        for i, res in enumerate(aa_list_full):
            res_name = res.resname
            res_num = res_num+1
            for j in range(3):
                serial_num= serial_num +1
                x,y,z = c[3*i+j]
                if j == 0:
                    atom_name = 'N'
                if j == 1:
                    atom_name = 'CA'
                if j == 2:
                    atom_name = 'C'
                last2 = 1
                last1 = 0
                outfile.write( "ATOM%7d%5s%4s%2s%4d%12.3f%8.3f%8.3f%6.2f%6.2f\n" % 
                (serial_num, atom_name, res_name ,chain, res_num, x, y, z,last1, last2))
            serial_num= serial_num +1
            x,y,z = c2[i]
            atom_name = 'CB'
            last2 = 1
            last1 = 0
            outfile.write( "ATOM%7d%5s%4s%2s%4d%12.3f%8.3f%8.3f%6.2f%6.2f\n" % 
            (serial_num, atom_name, res_name ,chain, res_num, x, y, z, last1, last2))
            

def opi_pro(DEVICE, pdb_names,NATIVE_DCOY, PDB_PATH, FG_N, ITER,n,n_divide):
    #train_name = TRAIN[1]    
    #print(pdb_name,DEVICE)
    #train_name = TRAIN[num[1]]
    #for i in range()
    for i in tqdm(range(n*n_divide,(n+1)*n_divide)):
        
        pdb_name = NATIVE_DCOY+pdb_names[i]+"_model.pdb"
        native = NATIVE_DCOY +pdb_names[i]+"_native.pdb"
        path_decoy_result = path_opi_structure_pdb(path_r,pdb_names[i])
        #rmsd_file = path_decoy_result + "/rmsd.txt"
        print("***prtein:%d***" % i)
        #print(pdb_name,native)
        if pdb_name[-3:] == "npy":
            pdb_name, chain = get_pdb_name(PDB_PATH, pdb_name)
            aa_list_full = read_pdb(pdb_name, chain)

        elif pdb_name == "../example5/3fxiA_model.pdb":
            print("*"*20)
            aa_list_full = read_pdb(pdb_name,'A')
            
            #print(len(aa_list_full))
        else:
            #print(pdb_name)
            #pdb_name, chain = get_pdb_name(PDB_PATH, pdb_name)
            aa_list_full = read_pdb(pdb_name)
        
        ca_list, c_list, n_list= check_backbone_3(aa_list_full)
        cb_list = get_cb_list(aa_list_full)
        print("*"*20)
        if not ca_list or not c_list or not n_list:
            print("coord error!")
            sys.exit()

        backbone  = get_backbone(ca_list, c_list, n_list)
        mask = get_mask(ca_list)
        seq_list = get_seq(aa_list_full,t_dic)

        #inner_tensor = get_inner_coord(backbone)
        inner_tensor =c2inner_coord(backbone)
        inner_coord_cb_tensor = get_inner_coord_cb(ca_list, cb_list, n_list, c_list)
        mainchain_coord_tensor = torch.from_numpy(np.array([i.coord for i in backbone],dtype = 'float64'))
        ### net, lossfunction, optimizer###    
        net = NeRF_net(inner_tensor)
        net2 = NeRF_net_cb(inner_coord_cb_tensor) 
        net_pnerf = PNERF_net(inner_tensor)
        
        net = net.double()
        net2 = net2.double()
        net_pnerf = net_pnerf.double()
        
        criterion = nn.NLLLoss()
        if FG_N:
            print("PNERF...")
            optimizer = optim.SGD([net_pnerf.psi,net_pnerf.phy], lr=LR)
        else:
            optimizer = optim.SGD(net.dhd_v, lr=LR)   
        
        ###refinement###
        for i in range(ITER):
            optimizer.zero_grad()
            if FG_N:	
                c =net_pnerf(FG_N, mainchain_coord_tensor)
            else:
                c = net(mainchain_coord_tensor)
            #print(c.dtype)
            c2 = net2(c)
            c = c.to(DEVICE)
            c2 = c2.to(DEVICE)
            if opt.DEVICE=="cuda:0": 
                dist, angle, num_cs,cad,abb = get_feature_matrix(c, c2,DEVICE,WINDOW)
                #print(path_decoy_result)
                #print("native:",native)	
                save_opi_strcture(c,c2,aa_list_full,i,path_decoy_result)
                num_cs1 = num_cs.cpu().numpy()
            else:
                dist11, angle11, num_cs1 = get_feature(c,c2,DEVICE)
                dist = torch.cat(dist11).view(-1,16)
                angle_ = []
                for ii in range(len(angle11)):
                    x = torch.cat(angle11[ii])
                    angle_.append(x)
                    angle = torch.cat(angle_).view(-1,16,6)
                save_opi_strcture(c,c2,aa_list_full,i,path_decoy_result)
            #print(DEVICE)
            #print(num_cs1, mask)         
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
            opi_decoy = (path_decoy_result+"/%d.pdb" % (i+1))
            if native:
                rmsd = cal_rmsd_ca_new(opi_decoy,native)
                optimizer.step()
                time1 = time_cost(START)
                print(" iter_result:%d ,opi_loss %f time_cost%10.4fs RMSD:%.4f\n" % (i, loss.item(), time1, rmsd))
            else:
                optimizer.step()
                time1 = time_cost(START)
                print(" iter_result:%d ,opi_loss %f time_cost%10.4fs\n" % (i, loss.item(), time1))
        

def cal_rmsd_ca_new(decoy_name, native):
    print("name_list:",decoy_name, native)
    if native == "../example5/3fxiA_native.pdb":
        decoy_list = read_pdb(decoy_name,"A")
        native_list = read_pdb(native, "A")
    else:
        decoy_list = read_pdb(decoy_name)
        native_list = read_pdb(native)
    ca_num_decoy = [i.id[1] for i in decoy_list ]
    ca_num_native = [i.id[1] for i in native_list ]
    ca_num = [i for i in ca_num_decoy if i in ca_num_native]
    ca1 = []
    ca2 = []    
    for i in decoy_list:
        #if i.id[1] in ca_num:
            ca1.append(i['CA'].coord)
    for j in native_list:
        #if j.id[1] in ca_num:
            ca2.append(j['CA'].coord)
    x = np.array(ca1)
    y = np.array(ca2)
    #print(x[:30])
    #print(y[:30])
    sup = SVD.SVDSuperimposer()
    sup.set(x, y)
    sup.run()
    rms = sup.get_rms()
    return rms

def path_opi_structure_pdb(PATH,decoy_name):
    ''' get save opi_structure pdb path '''
    PATH1 = PATH +decoy_name+'_'
    isExists=os.path.exists(PATH1)
    if not isExists:
        os.makedirs(PATH1)
    return PATH1


import sys 
parser = argparse.ArgumentParser("protein stucture refinemnet..")
parser.add_argument('--ITER', type=int, default=50 )
parser.add_argument('--WINDOW', type=int, default=16) #Number of surrounding amino acids
parser.add_argument('--LR', type=float, default=0.05)#0.0005
parser.add_argument('--ENRAOPY_W', type=int, default=1)
parser.add_argument('--L1_smooth_parameter', type=float, default=10)#1.2
parser.add_argument('--EPOCH', type=int, default=1)
parser.add_argument('--PDB_PATH',  default="../pdb_/")
parser.add_argument('--NATIVE_DECOY',  default="../example5/")
parser.add_argument('--native_name',  default="3fxiA_native.pdb")#101m__native.pdb
parser.add_argument('--decoy_name', default="3fxiA_model.pdb")#101m__model.pdb
parser.add_argument('--DEVICE', default = "cuda:0") #cpu，cuda:0
parser.add_argument('--OPI', default = "SGD")
parser.add_argument('--FG', type =int, default = 0)
parser.add_argument('--PATH',  default="native_start")
parser.add_argument('--DATA_TYPE',  default="float64")
parser.add_argument('--PTH', default ='modelnb/best_seq_dist_angle_sincos16aa_tian0_gpu_770wei.pth')
parser.add_argument('--kn', type = int,default =10)
opt = parser.parse_args(args=[])
print("start...")
###############
###parameter###
###############
DATA = {}
ITER = opt.ITER
WINDOW = opt.WINDOW		
LR = opt.LR
EPOCH = opt.EPOCH
PDB_PATH = opt.PDB_PATH
DEVICE = torch.device(opt.DEVICE)
PTH = opt.PTH 
OPI = opt.OPI
#TRAIN=np.load('file/train_name_cb.npy')
#PDB_NAME = TRAIN[1]
#NATIVE_DCOY = opt.NATIVE_DECOY
NATIVE_DCOY = opt.NATIVE_DECOY
if NATIVE_DCOY == "../example4":
    ITER =6
print("INPUT PATH:",NATIVE_DCOY)

FG_N = int(opt.FG)
KN = opt.kn

result_path = "770_lr"+str(opt.LR)+"_sm"+str(opt.L1_smooth_parameter)+opt.OPI+"_entropy"+str(opt.ENRAOPY_W)+"/"
path_r = NATIVE_DCOY +opt.DEVICE[:4] + "_result/" + result_path

decoy_id = opt.PATH +"/"+opt.decoy_name  # opi decoy pdb
native = opt.PATH+'/'+opt.native_name


###model###    
#num=list(range(len(TRAIN)))    
model = Sim_p()
model = model.double()
model = model.to(DEVICE)    
model.load_state_dict(torch.load(PTH, map_location=DEVICE))
#g_data_net_cpu1 = g_data_net_cpu()	
g_data_net = g_data_net_tian0()
g_data_net = g_data_net.to(DEVICE)

####read data####
START = time.time()

#pdb_name = "4zgmA_all.npy"
#for pdb_name in list1[:1]:
#pdb_name = decoy_id
#pdb_names = os.listdir(PDB_PATH)

# for pdb_name in TRAIN[:2]:
#     #pdb_name = list1[0]
#     try:
#         opi_pro(pdb_name, PDB_PATH,FG_N,ITER,path_decoy_result,native=None)
#     except:
#         continue
native_decoy = os.walk(NATIVE_DCOY)

pdb_names = list(set([i[:5] for i in list(native_decoy)[0][2]]))
#########测试一个线程###########

#########################################


if __name__ == '__main__':
##############整个列表同时优化########################
# #     list1 = ["6cd7A_all.npy", "5idbA_all.npy",  "4zgmA_all.npy", "1m6sC_all.npy","2ozhA_all.npy","6edkA_all.npy"]
##list1 = ["4zgmA_all.npy", "6cd7A_all.npy", "5idbA_all.npy", "2yg9B_all.npy", "1m6sC_all.npy","2ozhA_all.npy","6edkA_all.npy","5gjhA_all.npy","3qbtH_all.npy","5x9kB_all.npy"]
#######################################
    print(DEVICE)
    print(pdb_names)
    n_divide = int(len(pdb_names)/KN)+1
    if len(pdb_names) == 1:        
        n = 0
        opi_pro(DEVICE,pdb_names,NATIVE_DCOY,PDB_PATH,FG_N,ITER,n,n_divide)
    else:
        torch.multiprocessing.set_start_method("spawn")
        for n in range(KN):
            p=Process(target=opi_pro,args=(DEVICE, pdb_names,NATIVE_DCOY,PDB_PATH,FG_N,ITER,n,n_divide,))
            p.start()
            
    time2 = time_cost(START)
    #print("Done, opi structure cost:%8.4fs\n" % time2)
        
