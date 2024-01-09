####################################################
####基于局部自由能地貌图的蛋白质结构优化软件#################
####################################################

1./public/software/apps/ghfund/ghfund202107012664文件夹下包括几个文件夹和文件说明：
1).example1, example2, example3, example4是四个算例。example1, example2, example3三个例子分别包含的是单个蛋白质算例，主要用来测试DCU和CPU计算结果的准确性。
 example4则是150个蛋白质的数据集，主要用来测试DCU的运行时间和对与CPU的加速比。
2). INSTALL是安装相关环境的文件夹:
	包含了miniconda、pytorch等安装文件   
3)refinment64_150是该软件的源代码文件夹:
	主程序gpu_cost.py, math_p.py是相关数学公式的模块，nn.py是神经网络模型模块，module蛋白质数据预处理模块。native_start文件夹存放的是输入文件，	modelnb存放的
    是训练好的神经网络模型，result存放优化后的蛋白质结构文件。
    

2.安装流程：
	该软件是基于pytorch和rocm4.0.1开发的，安装过程中配置环境需要安装anconda软件，创建虚拟环境和下载相关的python第三方库，
	具体包括如下几个方面（进入/public/software/apps/ghfund/ghfund202107012664/INSTALL/文件夹下进行安装）：
1).安装anconda软件包,命令行输入`Bash Miniconda3-py37_4.9.2-Linux-x86_64.sh`（Bash Miniconda3-py37_4.9.2-Linux-x86_64.sh文件已经下载）
2).创建虚拟环境，如：可命名环境为torch_t  输入`conda create --name torch_t python==3.6.13`，输入“conda activate torch_t”导入成功对话框前会显示“(torch_t)”
3).安装必要的包。如：numpy等包,这里可以从之前已经装好的环境中导出requirements.txt进行安装。输入`pip install -r requirements.txt`
4).使用pip安装torch的whl文件，` pip install torch-1.8.0+rocm4.0.1-cp36-cp36m-linux_x86_64.whl `

3.程序使用：
软件涉及的代码和脚本都放在了“/public/software/apps/ghfund/ghfund202107012664/refinement64_150”文件夹下
使用“sbatch run.sh”命令运行程序。程序结果在当前文件夹下的log文件，生成相应的pdb蛋白质结构文件在result文件夹下。

其中算例可以在run.sh中第14行和15行修改，例如计算第一个算例，命令行参数改为“..\example1”和“example1”即可。
最后输出在result.log文件中，包括了最终的a.正确结果:和b用时和加速比信息。与之前生成的result0.log文件可以进行比较。

默认的计算设备是DCU(单卡8核), 在gpu_cost.py 的287行可以对计算设备进行修改，如果使用CPU（单节点单核，example4是单节点8核）可将 默认参数 default = "cuda:0"改为default = "cpu"

已经运行的结果存放在result0.log 文件中，新运行的结果在result.log文件中，result.example1_cpu.log是example1在cpu下的结果。

4.其他
1).程序运行时间可能会有少许的波动，但是总体上在保证结果的前提下，使用单卡dcu与cpu相比 对example4中150个蛋白质的加速比保持在80倍以上。
2).gpu运行单个蛋白质加速情况只有10倍左右的原因是，一是启动dcu的时间较长，计算时间比较短（这个情况在example4中有明显的证实）。二是单个蛋白质优化的计算量不足以占满dcu的显存导致的计算浪费。
