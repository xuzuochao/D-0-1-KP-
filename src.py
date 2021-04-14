# coding=utf-8
import matplotlib.pyplot as plt
import linecache
import time
import datetime
import numpy as np

print("文件如下")
print("idkp1-10.txt\n""wdkp1-10.txt\n""udkp1-10.txt\n""sdkp1-10.txt")
tex=input("文件名称")
n=int(input("组号"))
m=n*8
s=n*8+2
t=n*8-2
profit=linecache.getline(tex,m)
list1 = profit.split(',')
weight=linecache.getline(tex,s)
list2 = weight.split(',')
list1.pop()
list2.pop()
#print("profit=",list1)
#print("weight=",list2)

numb=linecache.getline(tex,t)


list3=[]
list4=[]
list3=[int(x) for x in list1]
list4=[int(i) for i in list2]
def s1():
    weight=list4 
    profit=list3 
    plt.figure(figsize=(10, 10), dpi=100)
    plt.scatter(weight,profit)
    plt.show()
def s2():
    lit=[]
    for (a,b) in zip(list3,list4):
        num=a/b
        lit.append(num)
    lit= [round(i,3) for i in lit]
#print(lit)
    lit1=sorted(lit,reverse=True)
    print("降序",lit1)
    
print("\n",numb)





def s3():
    starttime = datetime.datetime.now()
    def bag_01(weights, values, capicity):
 
        n = len(values)
        f = [[0 for j in range(capicity+1)] for i in range(n+1)]
        for i in range(1, n+1):
            for j in range(1, capicity+1):
                f[i][j] = f[i-1][j]
                if j >= weights[i-1] and f[i][j] < f[i-1][j-weights[i-1]] + values[i-1]:
                    f[i][j] = f[i-1][j-weights[i-1]] + values[i-1]
        return f
 
    def show(capicity, weights, f):
        n = len(weights)
        print("最大价值:", f[n][capicity])
        x = [False for i in range(n)]
        j = capicity
        for i in range(n, 0, -1):
            if f[i][j] > f[i-1][j]:
                x[i-1] = True
                j -= weights[i-1]
        print("背包中所装物品为:")
        for i in range(n):
            if x[i]:
                print("第{}个,".format(i+1),end='')
    if __name__=='__main__':
        n=int(input("数量"))
        capicity=int(input("容量"))
        weights=list4
        values=list3
        m = bag_01(weights, values, capicity)
        show(capicity, weights, m)
      
        endtime = datetime.datetime.now()
        print ("\n\n\n\n                                运行时间",(endtime - starttime).seconds,"s")
def s4():
    start = time.time()
    def init(N,n):
        C = []
        for i in range(N):
            c = []
            for j in range(n):
                a = np.random.randint(0,2)
                c.append(a)
            C.append(c)
        return C


##评估函数
# x(i)取值为1表示被选中，取值为0表示未被选中
# w(i)表示各个分量的重量，v（i）表示各个分量的价值，w表示最大承受重量
    def fitness(C,N,n,W,V,w):
        S = []##用于存储被选中的下标
        F = []## 用于存放当前该个体的最大价值
        for i in range(N):
            s = []
            h  = 0  # 重量
            h=int(h)
            f  = 0  # 价值
            f=int(f)
            for j in range(n):
                if C[i][j]==1:
                    if h+W[j]<=w:
                        h=h+W[j]
                        f = f+V[j]
                        s.append(j)
            S.append(s)
            F.append(f)
        return S,F

##适应值函数,B位返回的种族的基因下标，y为返回的最大值
    def best_x(F,S,N):
        y = 0
        x = 0
        B = [0]*N
        for i in range(N):
            if y<F[i]:
                x = i
            y = F[x]
            B = S[x]
        return B,y

## 计算比率
    def rate(x):
        p = [0] * len(x)
        s = 0
        for i in x:
            s += i
        for i in range(len(x)):
            p[i] = x[i] / s
        return p

## 选择
    def chose(p, X, m, n):
        X1 = X
        r = np.random.rand(m)
        for i in range(m):
            k = 0
            for j in range(n):
                k = k + p[j]
                if r[i] <= k:
                    X1[i] = X[j]
                    break
        return X1

##交配
    def match(X, m, n, p):
        r = np.random.rand(m)
        k = [0] * m
        for i in range(m):
            if r[i] < p:
                k[i] = 1
        u = v = 0
        k[0] = k[0] = 0
        for i in range(m):
            if k[i]:
                if k[u] == 0:
                    u = i
                elif k[v] == 0:
                    v = i
            if k[u] and k[v]:
                # print(u,v)
                q = np.random.randint(n - 1)
                # print(q)
                for i in range(q + 1, n):
                    X[u][i], X[v][i] = X[v][i], X[u][i]
                k[u] = 0
                k[v] = 0
        return X
    
##变异
    def vari(X, m, n, p):
        for i in range(m):
            for j in range(n):
                q = np.random.rand()
                if q < p:
                    X[i][j] = np.random.randint(0,2)

        return X


    m = 8##规模
    N = int(input("数量"))  ##迭代次数
    Pc = 0.8 ##交配概率
    Pm = 0.05##变异概率
    V =list3
    W =list4
    n = len(W)##染色体长度
    w = int(input("容量"))

    C = init(m, n)
    S,F  = fitness(C,m,n,W,V,w)
    B ,y = best_x(F,S,m)
    Y =[y]
    for i in range(N):
        p = rate(F)
        C = chose(p, C, m, n)
        C = match(C, m, n, Pc)
        C = vari(C, m, n, Pm)
        S, F = fitness(C, m, n, W, V, w)
        B1, y1 = best_x(F, S, m)
        if y1 > y:
            y = y1
        Y.append(y)
    print("最大值为：",y)
    end = time.time()
    print("共耗时:" + str(end - start) + " s")
    plt.plot(Y)
    plt.show()
    
def s5():
    def backpack(number, weight, w, v):
    #初始化二维数组，用于记录背包中个数为i，重量为j时能获得的最大价值
        result = [[0 for i in range(weight+1)] for i in range(number+1)]
    #循环将数组进行填充
        for i in range(1, number+1):
            for j in range(1, weight+1):
                if j < w[i-1]:
                    result[i][j] = result[i-1][j]
                else:
                    result[i][j] = max(result[i-1][j], result[i-1][j-w[i-1]] + v[i-1])
        return result


    def main():
        number = int(input("数量"))
        weight = int(input("容量"))
        w = list4
        v = list3
        start = time.time()
        result = backpack(number, weight, w, v)
        end = time.time()
        print("共耗时:" + str(end - start) + " s")
        print("最优解为：" + str(result[number][weight]) + "\n")
        print("所选取的物品为：")
        item = [0 for i in range(number+1)]
        j = weight
        for i in range(1, number+1):
            if result[i][j] > result[i-1][j]:
                item[i-1] = 1
                j -= w[i-1]
        for i in range(number):
            if item[i] == 1:
                print("第" + str(i+1) + "件",end="")
    if __name__ == '__main__':
        main()


for abc in range(10):
    print("\n\n------------------------------------------------------------------------------------------------")
    print("|                                          1,画散点图                                          |")
    print("|                                          2,降序排序                                          |")
    print("|                                          3,回溯法求解                                        |")
    print("|                                          4,遗传算法求解                                      |")
    print("|                                          5,动态规划求解                                      |")
    print("------------------------------------------------------------------------------------------------\n\n")
    sss=int(input("编号"))
    if sss==1: s1()
    if sss==2: s2()
    if sss==3: s3()
    if sss==4: s4()
    if sss==5: s5()

