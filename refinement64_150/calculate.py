import sys
#print(len(sys.argv))
#print(sys.argv)
example = sys.argv[1]
#print(example)
file_cpu = "/public/software/apps/ghfund/ghfund202107012664/"+example+"/cpu_result/result_cpu.log"
file_dcu = "result.log"
def read(file_name,example):
    with open(file_name,"r") as t:
        lines = t.readlines()
        line = lines[-2:][0]
        line = line.strip('\n')
        list1 = line.split()
        if example == 'example4':
            #print('*'*10)
            #print(line)
            print("时间：%8.4fs " % float(list1[4][-10:-1]))
        else:
            print(line)
        #print(list1)
        time = float(list1[4][-10:-1] )
        rmsd = float(list1[5][5:])
        loss = float(list1[2])
    return rmsd,time,loss
print("CPU单线程计算结果:")
t1 = read(file_cpu,example)
print("DCU单卡计算结果:")
t2 = read(file_dcu,example)
rmsd_c, time_c, loss_c = t1
rmsd_d, time_d, loss_d = t2
RMSD_error = (rmsd_c-rmsd_d)/rmsd_c
LOSS_error = (loss_c-loss_d)/loss_c
nnn = time_c/time_d
if example == 'example4':
    print("*"*20)
    print("b.用时:%8.4fs,加速比%8.4f倍" % (time_d, nnn))
else:
    print("*"*20)
    print("a.正确结果:loss误差:%.4f%%, RMSD误差:%0.4f%%\n" % ( RMSD_error,LOSS_error))
    print("*"*20)
    print("b.用时:%8.4fs,加速比%8.4f倍" % (time_d, nnn))
