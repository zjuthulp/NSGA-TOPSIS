import geatpy as ea
import numpy as np
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



class MyProblem(ea.Problem):   # 继承Problem父类
    def __init__(self):
        name = 'NSGAII-RTF'    # 初始化name
        M = 4  # 优化目标个数
        maxormins = [1,1,-1,-1]     # 初始化maxorimins（目标最小化取1）
        Dim = 10   # 初始化Dim (决策变量维数)
        varTypes = [0,0,0,0,0,0,0,0,0,0]   # 初始化varTypes（决策变量的类型，实数：0，整数：1）
        lb = [390,0,0,0,500,0.5,0,0,0,0.35]
        ub = [1000,110,300,200,1000,1.75,7.5,4.95,0.5,0.52]

        lbin = [1,1,1,1 ,1,1,1,1,1 ,1]        # 决策变量下边界（1表示包含该变量的下边界）
        ubin = [1,1,1,1 ,1,1,1,1,1 ,1]
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins,
                            Dim, varTypes, lb, ub, lbin, ubin)


        
    def evalVars(self, Vars):   # 目标函数
        # 参数命名
        #参数设计
        # OPC
        OPC=Vars[:, [0]]
        # SAC
        SAC = Vars[:, [1]]
        # SF
        SF = Vars[:, [2]]
        # FA
        FA = Vars[:, [3]]
        # S
        S = Vars[:, [4]]
        # MAXSS
        MAXSS = Vars[:, [5]]
        # TA
        TA = Vars[:, [6]]
        # ESA
        ESA = Vars[:, [7]]
        # SP / B
        SP_B = Vars[:, [8]]
        # W / B
        W_B = Vars[:, [9]]


        ####单位成本
        OPC_cost = 0.136
        SAC_cost = 0.434
        SF_cost = 2.052
        FA_cost = 3.137
        S_cost = 1.277
        TA_cost = 4.964
        ESA_cost = 7.585
        SP_B_cost = 9.515
        W_B_cost = 0.00039

        # 单位质量碳排放量
        OPC_CE = 0.931
        SAC_CE = 0.523
        SF_CE = 0.020
        FA_CE = 0.017
        S_CE = 0.0026
        TA_CE = 0.011
        ESA_CE = 0.013
        SP_B_CE = 0.250
        W_B_CE = 0.000196


        # 第一个目标函数，成本
        Cost=OPC_cost*OPC+SAC_cost*SAC+SF_cost*SF+FA_cost* FA+S_cost *S+TA_cost *TA+ESA_cost *ESA+SP_B_cost *SP_B+W_B_cost *W_B

        # 第二个目标函数，碳排放
        CE_cost = OPC_CE * OPC + SAC_CE * SAC + SF_CE * SF + FA_CE * FA + S_CE * S + TA_CE * TA + ESA_CE * ESA + SP_B_CE * SP_B + W_B_CE * W_B

        # 第三个目标函数，动态屈服应力
        # 读取数据
        data = pd.read_excel('动态屈服应力test.xlsx')
        # 分割自变量和目标变量
        X = data.iloc[:, 0:10]
        y = data.iloc[:, 10]
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        # 训练模型
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        dontai=0
        for i, feature in enumerate(X.columns):
            dontai=dontai+rf.feature_importances_[i]*Vars[:, [i]]
        # 指定目标
        dontai=np.abs(80-dontai)

        # 第四个目标函数，塑性粘度
        # 读取数据
        data = pd.read_excel('塑性粘度test.xlsx')
        # 分割自变量和目标变量
        X = data.iloc[:, 0:10]
        y = data.iloc[:, 10]
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        # 训练模型
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        nianxing = 0
        for i, feature in enumerate(X.columns):
            nianxing = nianxing + rf.feature_importances_[i] * Vars[:, [i]]

        # 指定目标
        nianxing=np.abs(2.5-nianxing)

        ObjV = np.hstack([Cost,CE_cost,dontai,nianxing])


        #约束

        #比例约束
        CV1 = []  # 转换成≤0
        for i in range(len(OPC)):
            CV1.append(SF[i]/(OPC[i]+SAC[i]+SF[i]+FA[i])-0.6)
        CV2 = []
        for i in range(len(OPC)):
            CV2.append(-(SF[i] / (OPC[i] + SAC[i] + SF[i] + FA[i])))

        CV3 = []
        for i in range(len(OPC)):
            CV3.append(FA[i] / (OPC[i] + SAC[i] + SF[i] + FA[i]) - 0.6)
        CV4 = []
        for i in range(len(OPC)):
            CV4.append(-(FA[i] / (OPC[i] + SAC[i] + SF[i] + FA[i])))

        CV5 = []
        for i in range(len(OPC)):
            CV5.append(S[i] / (OPC[i] + SAC[i] + SF[i] + FA[i]) - 2)
        CV6 = []
        for i in range(len(OPC)):
            CV6.append(0.5-(S[i] / (OPC[i] + SAC[i] + SF[i] + FA[i])))

        CV7 = []
        for i in range(len(OPC)):
            CV7.append(SP_B[i] / (OPC[i] + SAC[i] + SF[i] + FA[i]) - 0.5)
        CV8 = []
        for i in range(len(OPC)):
            CV8.append(-(SP_B[i] / (OPC[i] + SAC[i] + SF[i] + FA[i])))

        CV9 = []
        for i in range(len(OPC)):
            CV9.append(SP_B[i] / (OPC[i] + SAC[i] + SF[i] + FA[i]) - 0.52)
        CV10 = []
        for i in range(len(OPC)):
            CV10.append(0.3-(SP_B[i] / (OPC[i] + SAC[i] + SF[i] + FA[i])))

        #体积约束
        OPC_U=3150
        SAC_U=3050
        SF_U=1600
        FA_U=2400
        S_U=1400
        TA_U=2200
        ESA_U=1150
        SP_U=1100
        W_U=1000

        Vm=[]
        CV11 = []
        for i in range(len(OPC)):
            CV11.append(OPC[i]/OPC_U+SAC[i]/SAC_U+SF[i]/SF_U+FA[i]/FA_U+S[i]/S_U+TA[i]/TA_U+ESA[i]/ESA_U+SP_B[i]/SP_U+W_B[i]/W_U-1)

        CV12 = []
        for i in range(len(OPC)):
            CV12.append(
                1-(OPC[i] / OPC_U + SAC[i] / SAC_U + SF[i] / SF_U + FA[i] / FA_U+ S[i] / S_U + TA[i] /TA_U +ESA[i] / ESA_U+ SP_B[i] / SP_U + W_B[i] / W_U) )


        Cv_sum=[CV1, CV2, CV3, CV4, CV5, CV6, CV7, CV8, CV9,CV10, CV11, CV12]
        Cv_sum.pop()  # 去除最后一个元素 CV12
        Cv_sum.pop(-2)  # 去除倒数第二个元素 CV10

        # for t in range(10):
        #     CV.append(H[t]-140)
        #     CV.append(117 -


        '''
       <class 'numpy.ndarray'>
       [[-10]
         [-13]
         [ -2]
         [ -9]
         [ -5]]
        '''
        CV = np.hstack(Cv_sum)
        # print(CV)



        return ObjV,CV




#实例化问题对象
problem  = MyProblem()
# 构建智能优化算法
algorithm = ea.moea_NSGA2_templet(problem,
                                  # RI编码， 种群个体数量
                                  ea.Population(Encoding='RI', NIND=50),
                                  MAXGEN=50,  # 最大进化代数
                                  FitnV=1*10**(-4),  # 适应度函数偏差
                                  logTras=1,
                                  drawing=1,
                                  verbose=True)   # 表示每隔多少代记录一次日志信息，0表示不记录

# algorithm.drawing=2

# 求解函数
res = ea.optimize(algorithm, seed=1, verbose=True, drawing=1, outputMsg=False,
                  drawLog=False, saveFlag=True, dirName='result')
print(res)

# 调用run执行算法，得到帕累托最优解集NDSet
NDSet,pop = algorithm.run()
A=str(NDSet[0]).replace('{','').replace('}','')
# print(pop[99])
print('结果',str(NDSet[0]).replace('{','').replace('}',''))


# print('结果',NDSet[0].get('Population ObjV'))
NDSet.save()         #  把结果保存到文件中

# 输出
print('用时：%f秒'%(algorithm.passTime))
print('评价次数：%d次'%(algorithm.evalsNum))
print('非支配个体数：%d个'%(NDSet.sizes))
print('单位时间找到帕累托前沿点个数：%d个'%(int(NDSet.sizes //algorithm.passTime)))
print('目标结果值：', NDSet.ObjV)

PF = problem.getReferObjV()

if PF is not None and NDSet.sizes != 0:
    GD = ea.indicator.GD(NDSet.Objv, PF)  # 计算GD指标
    IGD = ea.indicator.IGD(NDSet.Objv, PF)  # 计算IGD指标
    HV = ea.indicator.HV(NDSet.Objv, PF)   # 计算HV指标
    Spacing = ea.indicator.Spacing(NDSet.Objv, PF)  # 计算Spacing指标
    print('GD:%f'%GD)
    print('IGD:%f'%IGD)
    print('HV:%f'%HV)
    print('Spacing:%f'%Spacing)

if PF is None:
    metricName = [['IGD'], ['hv']]
    # [NDSet_trace, Metrics] =ea.indicator.moea_tracking(algorithm.pop_trace, PF,
    #                                                    metricName, problem.maxormins)
    # ea.trcplot(Metrics, labels = metricName, titles = metricName)


#
# TOPSIS分析
data1=NDSet.ObjV
#计算行数和列数
[m,n]=data1.shape
#print('行数：',m)
#print('列数：',n)
#数据标准化
data2=data1.astype('float')
for j in range(0,n):
    data2[:,j]=data1[:,j]/np.sqrt(sum(np.square(data1[:,j])))
#print(data2)
#计算信息熵
p=data2
for j in range(0,n):
    p[:,j]=data2[:,j]/sum(data2[:,j])
#print(p)
E=data2[0,:]
for j in range(0,n):
    E[j]=-1/np.log(m)*sum(p[:,j]*np.log(p[:,j]))
#print(E)
# 计算权重
w=(1-E)/sum(1-E)
#print(w)
#得到加权后的数据
R=data2*w
#得到最大值最小值距离
r_max=np.max(R, axis=0)  #每个指标的最大值
r_min=np.min(R,axis=0)   #每个指标的最小值
d_z = np.sqrt(np.sum(np.square((R -np.tile(r_max,(m,1)))),axis=1))  #d+向量
d_f = np.sqrt(np.sum(np.square((R -np.tile(r_min,(m,1)))),axis=1))  #d-向量
#得到评分
s=d_f/(d_z+d_f )
max_value = np.max(s)
s = np.delete(s, np.where(s == max_value))
Score=s/max(s)
for i in range(0,len(Score)-1):
    print(f"第{i+2}个结果得分为：{Score[i+1]}")