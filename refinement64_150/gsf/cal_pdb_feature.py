# import Bio
# from Bio import PDB
# from Bio.PDB.PDBParser import PDBParser
# from Bio.PDB.DSSP import DSSP
from pre_data_old import *
import numpy as np
import threading
from multiprocessing import Pool
from tqdm import tqdm
t_dic={'ALA':'A','VAL':'V','LEU':'L','ILE':'I','PHE':'F','TRP':'W','MET':'M','PRO':'P',\
       'GLY':'G','SER':'S','THR':'T','CYS':'C','TYR':'Y','ASN':'N','GLN':'Q','HIS':'H',\
       'LYS':'K','ARG':'R','ASP':'D','GLU':'E'}
path = "../pdb_/"
#path = "/state/partition1/cxy/pdb_xray_127027_protein/"
pdb_list_file = "cullpdb_pc25_res2.0_R0.25_d181126_chains9311"
#pdb_list_file = "/state/partition1/cxy/cullpdb_pc90_res3.0_R1.0_d191107_chains39689"
#pdb_list_file = "/state/partition1/cxy/cullpdb_pc50_res2.0_R0.25_d191010_chains17543"
NB_NUMAA = 24 

class  MyThread(threading.Thread):
    def __init__(self, func, args, name =''):
        threading.Thread.__init__(self)
        self.name = name
        self.func = func
        self.args = args

    def getResult(self):
        return self.res

    def run(self):
        print("starting...", self.name)
        self.res = self.func(*self.args)
        print(self.name, "finished...")


if __name__ == "__main__":
    def main1(NB_NUMAA, n):
        pdb_id, pdb_chain = get_id_chain_name(pdb_list_file)
        num = list(range(len(pdb_id)))
        #print(len(num))
        #for i in num[n*20:(n+1)*20]:  #n is the number of subprocesses
        for i in tqdm(num[:]):  #n is the number of subprocesses
            #print(pdb_id[i])
            if len(pdb_id[i]) !=4:
                continue
            pdb_name=path + "pdb"+pdb_id[i].lower()+'.ent'
            pdb_name= pdb_name.replace('\\','/')
            print(pdb_name)
            #pdb_name = path+pdb_id[i].lower()+".pdb.gz"
            chain = pdb_chain[i]

            #print(pdb_name)
            print("reading %s..." % pdb_name)
            #try:
            aa_list_full = read_pdb(pdb_name, chain)
                #ss = get_dssp(pdb_name,chain)
            #except:
                #print("read %s fail " % pdb_name)
                #continue
            if not aa_list_full:
                continue
            seq_list = get_seq(aa_list_full)
            #if len(ss) != len(seq_list):
                #print(len(ss), len(seq_list))
                #continue
            ca_list =get_atom_list_npy(aa_list_full,'CA')
            cb_list = get_atom_list_npy(aa_list_full,'CB')
            c_list = get_atom_list_npy(aa_list_full,'C')
            n_list = get_atom_list_npy(aa_list_full,'N')

            #dps = cal_depth(s, aa_list_full)
            #hse_a, hse_b = cal_hseab(s, aa_list_full)
            #pssm = get_pssm(pssm_train_data,pssm_test_data)
            ca_dist = cal_dist(ca_list)
            mask = get_mask(ca_list)

            ids=ca_dist==None#?????????????
            ca_dist[ids]=100   #算不出来距离的设置为100
            ca_dist_cs=[]
            angle_cs=[]
            num_cs=[]
            #pssm_cs = []
            for j in range(len(ca_dist)):
                t = ca_dist[j]
                s=t.argsort()
                aa_num24 = s[1:NB_NUMAA+1]
                ca_dist_cs.append(t[s[1:NB_NUMAA+1]])
                angle_d = get_angle5_ceshi(aa_num24,ca_list, cb_list, n_list, c_list,j)
                angle_d = np.array(list(angle_d))
                angle_cs.append(angle_d)
                #angle_cs.append(angle_d[j][s[1:17]])
                #print(angle_d[j][s[1:17]])
                num_cs.append(s[1:NB_NUMAA+1])
            dic_r={}
            dic_r['dis']=ca_dist_cs #距离
            dic_r['adj_dis'] = ca_dist #完整的相邻距离矩阵
            dic_r['angle']=angle_cs #角度
            dic_r['mask']=mask      #ca
            dic_r['ids']=num_cs    #
            dic_r['seq']=seq_list  #序列
            #dic_r['dssp'] = ss
            #dic_r['dps']=dps        #氨基酸深度
            #dic_r['hsea']=hse_a     #裸球暴露面积
            #dic_r['hseb']=hse_b

            out_name='../pdb_other_cb_24aa/'+pdb_id[i].lower()+pdb_chain[i]+'_all.npy'
            np.save(out_name,dic_r)
            print("%d/%d-- cal finish!" % (i,len(pdb_id)))
    main1(NB_NUMAA,0)
    #threads =[]
    #for j in range(100):
        #name1 =("thread %s..." % ((j+1)*10))
        #t = MyThread(main, (NB_NUMAA,j), name1)
        #threads.append(t)
    #for j in range(100):
        #threads[j].start()
    #print ("Exiting Main Thread")
    
    # p = Pool(5)
    # for n in range(5):
    #     p.apply_async(main1, args=(NB_NUMAA,n))
    # print('Waiting for all subprocesses done...')
    # p.close()
    # p.join()
    # print('All subprocesses done.')


   
