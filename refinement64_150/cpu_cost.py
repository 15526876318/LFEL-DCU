
#!/usr/bin/python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import time
import argparse
import sys
import os 
#from gsf.pre_data import 
from module import get_feature,get_pdb_name,read_pdb,check_backbone_3,get_cb_list,get_backbone,get_mask,get_seq,get_inner_coord,\
        get_inner_coord_cb,get_feature_cpu
from nn import NeRF_net_cb, NeRF_net, g_data_net_cpu,Sim_p,g_data_net_tian0
import Bio.SVDSuperimposer as SVD
#from gpu_cost import generate1

def time_cost(start):
    time_cost = time.time() - start
    time_cost = round(time_cost, 4)
    return time_cost
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

def cal_rmsd_ca_new(decoy_name, native):
    decoy_list = read_pdb(decoy_name)
    native_list = read_pdb(native)
    ca_num_decoy = [i.id[1] for i in decoy_list ]
    ca_num_native = [i.id[1] for i in native_list ]
    ca_num = [i for i in ca_num_decoy if i in ca_num_native]
    ca1 = []
    ca2 = []    
    for i in decoy_list:
        if i.id[1] in ca_num:
            ca1.append(i['CA'].coord)
    for j in native_list:
        if j.id[1] in ca_num:
            ca2.append(j['CA'].coord)
    x = np.array(ca1)
    y = np.array(ca2)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser("protein stucture refinemnet..")
    parser.add_argument('--ITER', type=int, default=6 )
    parser.add_argument('--WINDOW', type=int, default=16) #Number of surrounding amino acids
    parser.add_argument('--LR', type=float, default=0.0005)
    parser.add_argument('--EPOCH', type=int, default=1)
    parser.add_argument('--PDB_PATH',  default="../pdb_/")
    parser.add_argument('--DEVICE', default = "cpu")
    parser.add_argument('--OPI', default = "SGD")
    parser.add_argument('--ENRAOPY_W', type=int, default=1)
    parser.add_argument('--L1_smooth_parameter', type=float, default=1.2)
    #parser.add_argument('--PTH', default ='../modelnb/best_seq_dis_angle.pth')
    parser.add_argument('--PTH', default ='../modelnb/best_seq_dist_angle_sincos16aa_tian0_gpu_770wei.pth')
    parser.add_argument('--NATIVE_DCOY',  default="./native_start")
    opt = parser.parse_args(args=[])
    print("start...")
	###parameter###
    ITER = opt.ITER
    WINDOW = opt.WINDOW		
    LR = opt.LR
    EPOCH = opt.EPOCH
    PDB_PATH = opt.PDB_PATH
    DEVICE = torch.device(opt.DEVICE)
    PTH = opt.PTH
    OPI = opt.OPI
    NATIVE_DCOY = opt.NATIVE_DCOY
    TRAIN=np.load('file/train_name_cb.npy')
    result_path = "770_lr"+str(opt.LR)+"_sm"+str(opt.L1_smooth_parameter)+opt.OPI+"_entropy"+str(opt.ENRAOPY_W)+"/"
    path_r = "result/" + result_path

	###model###    
    num=list(range(len(TRAIN)))    
    model = Sim_p()
    model = model.double()
    model = model.to(DEVICE)    
    model.load_state_dict(torch.load(PTH, map_location=DEVICE))
    g_data_net_cpu1 = g_data_net_cpu()	
    g_data_net = g_data_net_tian0()
    list1 = ["4zgmA_all.npy","6cd7A_all.npy", "5idbA_all.npy",   
    "1m6sC_all.npy","2ozhA_all.npy","6edkA_all.npy"]
    native_decoy = os.listdir(NATIVE_DCOY)
    pdb_names = list(set([i[:5] for i in native_decoy]))
    ####read data
    START = time.time()
    print(DEVICE)
    for i in range(len(pdb_names[:])):
        pdb_name = NATIVE_DCOY+"/"+pdb_names[i]+"_model.pdb"
        native = NATIVE_DCOY+"/" +pdb_names[i]+"_native.pdb"
        path_decoy_result = path_opi_structure_pdb(path_r,pdb_names[i])
        #train_name = list1[i]
        train_name  =  pdb_name   
        print(i, train_name)
        
        
        if pdb_name[-3:] == "npy":
            pdb_name, chain = get_pdb_name(PDB_PATH, pdb_name)
            aa_list_full = read_pdb(pdb_name, chain)
        else:
            #print(pdb_name)
            aa_list_full = read_pdb(pdb_name)
        #train_name = TRAIN[num[1]]
        ca_list, c_list, n_list= check_backbone_3(aa_list_full)
        cb_list = get_cb_list(aa_list_full)
        if None in  ca_list or None in  c_list or None in  n_list:
            print("coord error!")
            continue
            sys.exit()

        backbone  = get_backbone(ca_list, c_list, n_list)
        mask = get_mask(ca_list)
        seq_list = get_seq(aa_list_full,t_dic)
    
        inner_tensor = get_inner_coord(backbone)
        inner_coord_cb_tensor = get_inner_coord_cb(ca_list, cb_list, n_list, c_list)
        mainchain_coord_tensor = torch.from_numpy(np.array([i.coord for i in backbone],dtype = 'float64'))

        ### net, lossfunction, optimizer###    
        net = NeRF_net(inner_tensor)
        #net_pnerf = PNERF_net(inner_tensor)
        net2 = NeRF_net_cb(inner_coord_cb_tensor) 
        
        criterion = nn.NLLLoss()
        #optimizer = optim.SGD(net_pnerf.dhd_list, lr=LR)
        optimizer = optim.SGD(net.dhd_v, lr=LR)
        #optimizer_w = optim.SGD(model.parameters(), lr=0.1)    
        
        ###refinement###
        for i in range(6):
            optimizer.zero_grad()
            c = net(mainchain_coord_tensor)	
            c2 = net2(c)
            dist, angle, num_cs = get_feature(c,c2,DEVICE)
            dist11 = torch.cat(dist).view(-1,16)
            angle_ = []
            for ii in range(len(angle)):
                x = torch.cat(angle[ii])
                angle_.append(x)
                angle11 = torch.cat(angle_).view(-1,16,6)
            idx_t, y, kkn, index_t, index_h = generate1(mask, num_cs, seq_list) #
            x = g_data_net(dist11, angle11, idx_t, index_t, index_h,DEVICE)  
            outputs = model(x)
            #
            # x, idx_, dis_, angle_t, y, kkn = g_data_net_cpu1(mask, num_cs , dist, angle, seq_list,seqdic)
            # x=x.to(DEVICE)
            # dis_=dis_.to(DEVICE)
            # idx_=idx_.to(DEVICE)
            # angle_t=angle_t.to(DEVICE)
            #outputs=model(x,idx_,dis_,angle_t)

            #dist, angle, num_cs, ca_dist, ab = get_feature_cpu(c,c2,DEVICE,WINDOW)####
            #x, idx_, dis_, angle_t, y, kkn = g_data_net_cpu1(mask, num_cs , dist, angle, seq_list,seqdic)
            #outputs=model(x,idx_,dis_,angle_t)
            # idx_t, label, kkn, index_t, index_h = g_data(mask, num_cs, seq_list)
            # x = g_data_net(dist11, angle11, idx_t, index_t, index_h, DEVICE)




            
            loss = criterion(F.log_softmax(outputs,dim=1), y)
            loss.backward()
            
            opi_decoy = (path_decoy_result+"/%d.pdb" % (i+1))
            if native:
                rmsd = cal_rmsd_ca_new(opi_decoy,native)
                optimizer.step()
                time1 = time_cost(START)
                print(" iter:%d ,opi_loss %f time_cost%8.4fs RMSD:%.4f\n\n" % (i, loss.item(), time1, rmsd))
            else:
                optimizer.step()
                time1 = time_cost(START)
                print(" iter:%d ,opi_loss %f time_cost%8.4fs\n" % (i, loss.item(), time1))
    # time2 = time_cost(START)
    # print("Done, opi structure cost:%8.4fs\n" % time2)
        
