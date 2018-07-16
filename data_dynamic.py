#coding:utf-8
import math
#import talib
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
import random
import time
import talib
import itertools

count0 = 0
count1 = 0
count_1 = 0
count00 = 0
count_11 = 0
#总任务=建立数据集=把需要的数据建成一一对应的两个pkl文档，分别放入increaseX、increaseY
#生成所有txt数据集合，调用一个，生成一个
#filepath=txt路径，count,count_,=记录正例负例数目，inputF,inputS=一个说明哪些输入特征，一个是特征数目
#filepath包实现了兼容各操作系统的文件路径的实用操作函数
def buildDataForEachStock(filepath,count,count_,inputF,inputS):
    # 读表格，生成每个txt所有内容，df就是存放数据的地方，计算指标放入
    #col_index_num 指代的是你要显示的数据时你选中区域的第几列
    #sep应该是字符串类型的参数，在程序中的含义是“分隔符”。
    df = pd.read_table(filepath, names=['open', 'high', 'low', 'close', 'vol', 'value'], index_col=0, sep=' ')
    # 丢弃vol,value指标
    #如果我们调用 df.drop((name, axis=1),我们实际上删掉了一列，而不是一行，第0轴沿着行的垂直往下，第1轴沿着列的方向水平延伸
    df = df.drop(['vol', 'value'], axis=1)
    # 添加中间变量 便于查找某个日期对应的行数
    # np.arange()函数返回一个有终点和起点的固定步长的排列
    df['index'] = np.arange(len(df))
    '''
    增加macd和kdj（仅j线）的日、2日、周、月等四个周期的数据
    '''
    # MACD 1日 2日 一周 和一月
    df['MACD'], df['MACDsignal'], df['MACDhist_1'] = talib.MACD(df['close'].values,
                                                                fastperiod=12,
                                                                slowperiod=26,
                                                                signalperiod=9)
    df['MACD'], df['MACDsignal'], df['MACDhist_2'] = talib.MACD(df['close'].values,
                                                                fastperiod=24,
                                                                slowperiod=52,
                                                                signalperiod=18)
    df['MACD'], df['MACDsignal'], df['MACDhist_3'] = talib.MACD(df['close'].values,
                                                                fastperiod=36,
                                                                slowperiod=78,
                                                                signalperiod=27)
    df['MACD'], df['MACDsignal'], df['MACDhist_4'] = talib.MACD(df['close'].values,
                                                                fastperiod=48,
                                                                slowperiod=104,
                                                                signalperiod=36)
    df['MACD'], df['MACDsignal'], df['MACDhist_5'] = talib.MACD(df['close'].values,
                                                                fastperiod=60,
                                                                slowperiod=130,
                                                                signalperiod=45)

    # KDJ 1日 2日 1周 1月
    df['k'], df['d'] = talib.STOCH(df['high'].values, df['low'].values, df['close'].values,
                                   fastk_period=9, slowk_period=3, slowk_matype=0,
                                   slowd_period=3, slowd_matype=0)
    df['j_1'] = df['k'] * 3 - df['d'] * 2
    df['k'], df['d'] = talib.STOCH(df['high'].values, df['low'].values, df['close'].values,
                                   fastk_period=18, slowk_period=6, slowk_matype=0,
                                   slowd_period=6, slowd_matype=0)
    df['j_2'] = df['k'] * 3 - df['d'] * 2
    df['k'], df['d'] = talib.STOCH(df['high'].values, df['low'].values, df['close'].values,
                                   fastk_period=27, slowk_period=9, slowk_matype=0,
                                   slowd_period=9, slowd_matype=0)
    df['j_3'] = df['k'] * 3 - df['d'] * 2
    df['k'], df['d'] = talib.STOCH(df['high'].values, df['low'].values, df['close'].values,
                                   fastk_period=36, slowk_period=12, slowk_matype=0,
                                   slowd_period=12, slowd_matype=0)
    df['j_4'] = df['k'] * 3 - df['d'] * 2
    df['k'], df['d'] = talib.STOCH(df['high'].values, df['low'].values, df['close'].values,
                                   fastk_period=45, slowk_period=15, slowk_matype=0,
                                   slowd_period=15, slowd_matype=0)
    df['j_5'] = df['k'] * 3 - df['d'] * 2

    # 计算四个周期的cci
    df['cci_1'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
    df['cci_2'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values, timeperiod=28)
    df['cci_3'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values, timeperiod=42)
    df['cci_4'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values, timeperiod=56)
    df['cci_5'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values, timeperiod=70)
    # df['cci_20'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values, timeperiod=280)

    # 计算bias
    df['BIAS_1'] = (df['close'] - talib.MA(df['close'].values, timeperiod=6)) \
                   / talib.MA(df['close'].values, timeperiod=6) * 100
    df['BIAS_2'] = (df['close'] - talib.MA(df['close'].values, timeperiod=12)) \
                   / talib.MA(df['close'].values, timeperiod=12) * 100
    df['BIAS_3'] = (df['close'] - talib.MA(df['close'].values, timeperiod=18)) \
                   / talib.MA(df['close'].values, timeperiod=18) * 100
    df['BIAS_4'] = (df['close'] - talib.MA(df['close'].values, timeperiod=24)) \
                   / talib.MA(df['close'].values, timeperiod=24) * 100
    df['BIAS_5'] = (df['close'] - talib.MA(df['close'].values, timeperiod=30)) \
                   / talib.MA(df['close'].values, timeperiod=30) * 100

    # 计算cr
    df['MID'] = (df['high'] + df['low'] + df['close']) / 3
    df['REF_MID_1'] = df['MID'].shift(1)
    df['CR_UP'] = df['high'] - df['REF_MID_1']
    df.loc[df['CR_UP'] < 0, 'CR_UP'] = 0
    df['CR_DOWN'] = df['REF_MID_1'] - df['low']
    # df[df['CR_DOWN'] < 0] = 0
    df.loc[df['CR_DOWN'] < 0, 'CR_DOWN'] = 0
    df['CR_1'] = df['CR_UP'].rolling(window=26).sum() / df['CR_DOWN'].rolling(window=26).sum() * 100
    df['CR_2'] = df['CR_UP'].rolling(window=52).sum() / df['CR_DOWN'].rolling(window=52).sum() * 100
    df['CR_3'] = df['CR_UP'].rolling(window=78).sum() / df['CR_DOWN'].rolling(window=78).sum() * 100
    df['CR_4'] = df['CR_UP'].rolling(window=104).sum() / df['CR_DOWN'].rolling(window=104).sum() * 100
    df['CR_5'] = df['CR_UP'].rolling(window=130).sum() / df['CR_DOWN'].rolling(window=130).sum() * 100

    '''
    计算每一天涨幅
    计算公式：（当天收盘价-前1天的收盘价）/前1天的收盘价*100
    '''
    df['last_close'] = df['close'].shift(1)
    df['last_increase'] = (df['close'] - df['last_close'])/df['last_close']*100
    # 获取所有图形
    totalSeed = df.index.tolist()[1:]  # 第一天没有涨幅
    # 取出作为输入的指标
    df_new = df.loc[:,inputF]
    # 丢弃不会用到的中间指标
    df = df.drop(['MACDsignal', 'MACD', 'k', 'd',
                  'MID', 'REF_MID_1',
                  'CR_UP', 'CR_DOWN'],
                 axis=1)

    '''
    构建训练数据集
    '''
    #[ ]：代表list列表数据类型，列表是一种可变的序列
    trainingDataX = []
    trainingDataY = []
    global count0, count1, count_1
    global count00, count_11
    # 有空的数据都不要
    maxnull_index = df[df.isnull().values == True]['index'].max()
    #
    for eachindex in totalSeed[maxnull_index + 20:]:
        index = int(df.loc[eachindex, 'index'])
        # 注意iloc是前闭后开集合,0：20只取0-19。loc就根据这个index来索引对应的行，iloc是根据行号来索引，行号从0开始，逐次加1
        df_values = df_new.iloc[index - 19:index + 1, :]
        # 归一化 由于数据不是按高斯分布 所以按0-1进行线性变换
        df_values = (df_values - df_values.min()) / (df_values.max() - df_values.min())
        #
        increase = df.loc[eachindex,'last_increase']
        #
        jiaochapoints=jiaocha(eachindex,df)
        # if jiaochapoints.count(1)+jiaochapoints.count(2)>0:
        # 0-2,2-4,4-6,6-8,8-(下闭上开区间)
        # 采样,通过修改a>X中X的大小修改采样比重，X越大，采样越多，对采样的数据贴标签
        if increase > 0:
            if increase >= 8:
                trainingDataX.append(df_values.fillna(0).values)#将新图形加入到保存图形的集合中
                trainingDataY.append(5)
                if len(df_values) != 0:
                    count[4] += 1
            else:   #大于0小于等于8
                label = increase//2+1
                if label == 3:
                    a = random.randint(1, 100)
                    if a > 40:
                        continue
                if label == 2:
                    a = random.randint(1, 100)
                    if a > 15:
                        continue
                if label == 1:
                    a = random.randint(1, 100)
                    if a > 6:
                        continue
                # 加入训练数据集
                trainingDataX.append(df_values.fillna(0).values)
                #加入标签数据集
                trainingDataY.append(increase//2+1)
                # 统计数目
                if len(df_values) != 0:
                    count[int(increase//2)] += 1

        else:   #小于等于0
            if increase <= -8:
                trainingDataX.append(df_values.fillna(0).values)
                trainingDataY.append(0)
                if len(df_values) != 0:
                    count_[4] += 1
            else:   #大于0小于等于9
                label = abs(increase)//2+1
                if label == 3:
                    a = random.randint(1, 100)
                    if a > 40:
                        continue
                if label == 2:
                    a = random.randint(1, 100)
                    if a > 15:
                        continue
                if label == 1:
                    a = random.randint(1, 100)
                    if a > 6:
                        continue
                # 加入训练数据集
                trainingDataX.append(df_values.fillna(0).values)
                #加入标签数据集
                trainingDataY.append(-(abs(increase)//2+1))
                # 统计数目
                if len(df_values) != 0:
                    count_[int(abs(increase)//2)] += 1

    return trainingDataX, trainingDataY
def jiaocha(t,df_valueslist):
    # for t in range(1, len(df_valueslist['MACDhist_1'])):
    cols=['MACDhist_','j_','cci_','BIAS_','CR_']
    cross_points=[]
    for col in cols:
        colsnames=[]
        for j in range(1,6):
            colsnames.append(col+str(j))
        for i in itertools.combinations(colsnames, 2):
            # df.loc[eachindex, 'last_increase']
            ema_close_short=df_valueslist[i[0]]
            ema_close_long=df_valueslist[i[1]]
            tindex = int(df_valueslist.loc[t, 'index'])
            # df_values = df_valueslist.iloc[tindex-1 :tindex , :]
            #金叉
            if ema_close_short[t] > ema_close_short[tindex-1] and ema_close_short[t] > ema_close_long[t] \
                            and ema_close_short[tindex-1] < ema_close_long[tindex-1]:
                cross_points.append(1)
            #死叉
            elif ema_close_short[t] < ema_close_short[tindex-1] and ema_close_short[t] < ema_close_long[t] \
                            and ema_close_short[tindex-1] > ema_close_long[tindex-1]:
                cross_points.append(2)
            else:
                cross_points.append(0)
    return cross_points
def data_dynamic_do(inputF,inputS):
    inputsize = inputS
    start = time.clock()  #开始时间

    count = list(np.zeros(10))  #
    count_ = list(np.zeros(10))

    # inputsize = 29
    all_X = np.empty(shape=[0, 20, inputsize])  # 原始特征6+(macd,j,cci,bias,cr,vol_macd)*4个特征
    all_Y = np.empty(shape=[0, 1])

    # 首先清空历史数据
    for maindir, subdir, file_name_list in os.walk("./increaseX"):  #for循环检索结果文件
        for filename in file_name_list:
            #os.path.join()：  将多个路径组合后返回
            apath = os.path.join(maindir, filename)#获取文件路径
            os.remove(apath)#移除文件

    for maindir, subdir, file_name_list in os.walk("./increaseY"):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            os.remove(apath)

    # 读入新数据
    for maindir, subdir, file_name_list in os.walk("./train"):  # TXTDAY=201501-201701
        totalcount = 0
        totalfailcount = 0
        savecount = 0
        for filename in file_name_list:
            eachpath = os.path.join(maindir, filename)
            try:
                # 处理得到txt包含的图形
                trainingDataX, trainingDataY = buildDataForEachStock(eachpath,count,count_,inputF,inputS)
                if len(trainingDataX) == 0:
                    totalfailcount += 1
                    continue
                trainingDataY = np.array(trainingDataY)
                trainingDataY.shape = (len(trainingDataY), 1)
                # 保存进训练集
                all_X = np.vstack((all_X, trainingDataX))
                all_Y = np.vstack((all_Y, trainingDataY))
                print(filename + " is successfully processed")
                totalcount += 1
                # 每读取250个文件保存一次
                if totalcount % 250 == 0:
                    version = datetime.now().strftime("%m-%d-%H-%M")
                    model_name = "trainingDataX_increase"
                    model_name1 = "trainingDataY_increase"
                    result_file = "increaseX/{}_{}_time{}.pkl"
                    result_file1 = "increaseY/{}_{}_time{}.pkl"
                    # 保存成文件
                    with open(result_file.format(model_name, savecount, version), 'wb')  as f:
                        pickle.dump(all_X, f)
                    with open(result_file1.format(model_name1, savecount, version), 'wb')  as f:
                        pickle.dump(all_Y, f)
                    savecount += 1
                    # 再次初始化保存训练数据的集合
                    all_X = np.empty(shape=[0, 20, inputsize])  # 原始特征6+(macd,j,cci,bias,cr,vol_macd)*4=6+6*4-1=30-1个特征
                    all_Y = np.empty(shape=[0, 1])
            except:
                # 引发错误的原因是因为数据量不够
                print(filename + " has some mistakes")
                totalfailcount += 1
                continue

        # 将最后一批构建好的数据保存至本地文件中
        version = datetime.now().strftime("%m-%d-%H-%M")
        model_name = "trainingDataX_increase"
        model_name1 = "trainingDataY_increase"
        result_file = "increaseX/{}_{}_time{}.pkl"
        result_file1 = "increaseY/{}_{}_time{}.pkl"
        # 开始保存
        with open(result_file.format(model_name, savecount, version), 'wb')  as f:
            pickle.dump(all_X, f)
        with open(result_file1.format(model_name1, savecount, version), 'wb')  as f:
            pickle.dump(all_Y, f)
        # 计算耗时
        elapsed = (time.clock() - start)
        print("耗时:", elapsed)
        print("正10类数目为：\n", count)
        print("负10类数目为：\n", count_)
        print("读取成功的文件数", totalcount)
        print("读取为空的文件数", totalfailcount)
        print("finish")
        # 统计数据处理结果
        final_result = [count, count_, totalcount, totalfailcount]
        info_result = ['正10类数目为', '负10类数目为',
                       '读取成功的文件数', '读取为空的文件数']
        df_result = pd.DataFrame({"info": info_result, "count": final_result})
        info_file = "result/{}_time{}.csv"
        info_name = "data_analyse_increase"
        version = datetime.now().strftime("%m-%d-%H-%M")
        df_result.to_csv(info_file.format(info_name, version))
inputF=[ u'MACD', u'MACDsignal',
       u'MACDhist_1', u'MACDhist_2', u'MACDhist_3', u'MACDhist_4',
       u'MACDhist_5', u'k', u'd', u'j_1', u'j_2', u'j_3', u'j_4', u'j_5',
       u'cci_1', u'cci_2', u'cci_3', u'cci_4', u'cci_5', u'BIAS_1', u'BIAS_2',
       u'BIAS_3', u'BIAS_4', u'BIAS_5', u'MID', u'REF_MID_1', u'CR_UP',
       u'CR_DOWN', u'CR_1', u'CR_2', u'CR_3', u'CR_4', u'CR_5', u'last_close',
       u'last_increase']  #u'open', u'high', u'low', u'close', u'index',
inputS=6
data_dynamic_do(inputF,inputS)
