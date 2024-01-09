import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import copy
from simnetnb import Sim_p
import sys
from tqdm import tqdm
#import tracemalloc


seqdic={'A':0, 'R':1, 'D':2, 'C':3, 'Q':4, 'E':5, 'H':6, 'I':7, 'G':8, 'N':9, 'L':10, 'K':11, 'M':12, 'F':13, 'P':14, 'S':15, 'T':16, 'W':17, 'Y':18, 'V':19}

rootdir = '../pdb_other_cb_24aa'
#rootdir ='/state/partition1/cxy/pdb_other_chains17543_24aa/'
#tracemalloc.start()

def g_data(name,window):
    path = os.path.join(rootdir,name)
    path = path.replace('\\','/')
    data=np.load(path,allow_pickle=True).item()
    dis=np.array(data['dis'])[:,:window]
    angle=np.array(data['angle'])[:,:window]
    mask=data['mask']
    ids=np.array(data['ids'])[:,:window]
    seq=data['seq']
    idx_nb = []
    idx_unb = []
    label = []
    index_h=[]
    index_nb =[]
    index_unb = []
    kk=0
    for i in range(len(mask)):
        #nb_id = [j for j in range(-6+i,7+i) if j!=i]
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
                    for m in range(len(angle[i][j])):
                        if angle[i][j][m]==None:
                            angle[i][j][m]=random.random()*3.14
                    if seq[ids[i][j]] in seqdic:
                        idx_unb_i.append(seqdic[seq[ids[i][j]]])
                    else:
                        idx_unb_i.append(20)
            while len(idx_unb_i) < 10:
                index_unb_i.append(-1)
                idx_unb_i.append(21)
            for a in nb_id:
                if a in ids[i]:  #
                    k=np.where(ids[i]==a)
                    k=int(k[0][0]) #duole yiwei suoyixuyao suoyin liangci
                    index_nb_i.append(k)
                    for m1 in range(len(angle[i][k])):
                        if angle[i][k][m1]==None:
                            angle[i][k][m1]=random.random()*3.14
                    if seq[a] in seqdic:
                        idx_nb_i.append(seqdic[seq[a]])
                    else:
                        idx_nb_i.append(20)
                elif a == i:  #tianjia yucede anjisuan id 
                    index_nb_i.append(-1)
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

    label1 = torch.Tensor(label).long()
    index_h1 = torch.Tensor(index_h)
    angle1 = np.array(angle, dtype ='float32')
    dis1 = np.array(dis, dtype = 'float32')
    dis2 = torch.Tensor(dis1)
    angle2 = torch.Tensor(angle1)
    angle2_sin = torch.sin(angle2)
    angle2_cos = torch.cos(angle2)
    angle2_sin_cos = torch.cat((angle2_sin, angle2_cos), 2)
    return dis2, angle2_sin_cos, idx, label1, kk, index, index_h1

class g_data_net(nn.Module):
    def __init__(self):
        super(g_data_net, self).__init__()

    def forward(self,dist, angle, idx_t, index_t, index_h, device):

        dist_00 = torch.zeros_like(dist[:,0]).unsqueeze(1)
        angle_00 = torch.zeros_like(angle[:,0]).unsqueeze(1)
        dist_new = torch.cat((dist,dist_00), 1) # he angle_new yonglai tian 0
        angle_new = torch.cat((angle,angle_00),1)
        # print(idx_t.shape)
        h, w = idx_t.size()
        a = index_h.view(-1, 1)
        b = torch.ones(w,device = device).view(1, -1)
        #print(a.device, b.device)
        ab = torch.matmul(a, b).long()

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
    

if __name__ == "__main__":
    cuda = sys.argv[1]
    window = int(sys.argv[2])
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    model = Sim_p()
    model = model.to(device)
    g_data_net = g_data_net()
    g_data_net = g_data_net.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    train=np.load('../file/train_name_cb.npy')
    #train_val = os.listdir(rootdir)
    val=np.load('../file/val_name_cb.npy')
    test=np.load('../file/test_name_cb.npy')
    # print(val)
    #import os
    # train_val_test = []
    # for root, dirs, files in os.walk(rootdir):  
    #     for file in files:
    #         train_val_test.append(file)
    # train = train_val_test[:8000]
    # val = train_val_test[8000:8600]
    # test = train_val_test[8600:]
    num=list(range(len(train)))
    #train = num[:16000]
    #val = num[16000:]

    f_log=open('train_seq_dist_angle_sincos'+str(window)+'active_mish_770.log','w')

    epoches=30

    best_model_wts = None
    best_acc = 0.0
    print(device)
    for epoch in range(epoches):
        print('epoch:',epoch)
        model.train()
        random.shuffle(train)
        k=0
        kk=0
        running_loss = 0.0
        running_corrects = 0
        
        #train
        for i in tqdm(num[:]):
            #print(i,train[i])
            dist, angle,idx_t, y, kkn, index_t, index_h=g_data(train[i], window)
            dist = dist.to(device)
            angle=angle.to(device)
            idx_t=idx_t.to(device)
            index_t = index_t.to(device)
            index_h = index_h.to(device)
            y = y.to(device)
            x = g_data_net(dist, angle, idx_t, index_t,index_h,device)
            optimizer.zero_grad()
            outputs=model(x)
            loss = criterion(F.log_softmax(outputs,dim=1), y)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == y.data)
            running_loss += loss.item()
            #print(loss.item())
            k+=1
            kk+=kkn
            if (k-1)%1000==999:
                acc = running_corrects.double().cpu().data.numpy()/kk
                acc = round(acc,4)*100
                loss = round(running_loss/k,6)
                print(k,kk)
                print("loss:%f --- acc:%f " % (loss,acc))
        print('train:',epoch,running_loss/k,running_corrects.double().cpu().data.numpy()/kk,file=f_log)

        #val
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        k=0
        kk=0
        for i in tqdm(range(len(val))):
            #print(val[i])
            #if not g_data(val[i],window):
                #continue
            try:
                dist, angle, idx_t, y, kkn, index_t, index_h =g_data(val[i], window)
            except:
                continue

            dist = dist.to(device)
            angle=angle.to(device)
            idx_t=idx_t.to(device)
            index_t = index_t.to(device)
            index_h = index_h.to(device)
            y = y.to(device)
            x = g_data_net(dist, angle, idx_t, index_t,index_h,device)
            outputs=model(x)
            loss = criterion(F.log_softmax(outputs,dim=1), y)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == y.data)
            running_loss += loss.item()
            k+=1
            kk+=kkn
        print('val:',running_corrects.double().cpu().data.numpy()/kk)
        test_acc = running_corrects.double().cpu().data.numpy()/kk
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        print('val:',epoch,running_loss/k,running_corrects.double().cpu().data.numpy()/kk,file=f_log)
        
    #test
    model.load_state_dict(best_model_wts)
    running_loss = 0.0
    running_corrects = 0
    k=0
    kk=0
    for i in tqdm(range(len(test))):
        #if not g_data(test[i], window):
            #continue
        try:
            dist, angle, idx_t, y, kkn, index_t, index_h=g_data(test[i], window)
        except:
            continue
        dist = dist.to(device)
        angle=angle.to(device)
        idx_t=idx_t.to(device)
        index_t = index_t.to(device)
        index_h = index_h.to(device)
        y = y.to(device)
        x = g_data_net(dist, angle, idx_t, index_t,index_h,device)
        outputs=model(x)
        loss = criterion(F.log_softmax(outputs,dim=1), y)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == y.data)
        running_loss += loss.item()
        k+=1
        kk+=kkn
    print('test:',running_corrects.double().cpu().data.numpy()/kk)
    print('test:',epoch,running_loss/k,running_corrects.double().cpu().data.numpy()/kk,file=f_log)

    model.load_state_dict(best_model_wts)
    pth_file="modelnb/best_seq_770.pth"
    torch.save(model.state_dict(),pth_file)
