import numpy as np

class LogisticRegressionWithWeight:
    def __init__(self,learning_rate=7.0,iteration=2000,lam=0.1):
        self.learining_rate=learning_rate #学习率
        self.iteration=iteration   #迭代次数
        self.w=None
        self.err=1      #错误率
        self.b=0
        self.seed=None
        self.lam=lam    #正则化系数

    def sigmod(self,z):
        return 1.0/(1.0+np.exp(-z))

    def fit(self,train_data,train_label,sample_weight,epsilon=1e-8):
        feature_num=len(train_data[0])
        sample_num=len(train_label)
        np.random.seed(self.seed)
        self.w=np.random.normal(loc=0.0,scale=1.0,size=feature_num)
        self.b=np.random.normal(loc=0.0,scale=1.0)
        temp_h = self.sigmod(train_data.dot(self.w.T) + self.b)
        last_cost = -np.sum((train_label.T * np.log(temp_h) + (1 - train_label).T * np.log(1 - temp_h)) * sample_weight)\
                    +self.lam*self.w.dot(self.w.T)/(2*sample_num)
        for i in range(self.iteration):
            h=self.sigmod(train_data.dot(self.w.T)+self.b)
            self.w-=(self.learining_rate*(((sample_weight.T*(h-train_label.T)).T.dot(train_data)).T+self.lam*self.w/sample_num))
            self.b-=(self.learining_rate*(np.sum((h-train_label.T)*sample_weight)+self.lam*self.b/sample_num))
            now_cost = -np.sum((train_label.T * np.log(h + epsilon) + (1 - train_label.T) * np.log(1 - h + epsilon)) * sample_weight)\
                       +self.lam*self.w.dot(self.w.T)/(2*sample_num)
            if abs(now_cost-last_cost)<epsilon:
                break
            last_cost=now_cost
        temp_train_label=train_label*2-1
        self.err=np.sum((self.predict(train_data)!=temp_train_label.T)*sample_weight)
        return self

    def predict(self,test_data):
        return (self.sigmod(test_data.dot(self.w.T)+self.b)>=0.5)*2-1  #结果映射成-1和1




class DecisionTreeClassifierWithWeight:
    def __init__(self):
        self.err = 1  # 最小的加权错误率
        self.best_fea_index = 0  # 最优特征
        self.best_thres = 0  # 最优阈值
        self.best_op=1 #最优符号，1为>,0为<


    def fit(self, train_data, train_label, sample_weight):
        feature_num = train_data.shape[1]
        for i in range(feature_num):
            feature = train_data[:, i]  # 选定特征列
            fea_unique = np.sort(np.unique(feature))  # 将特征值排序
            for j in range(len(fea_unique) - 1):
                thres = (fea_unique[j] + fea_unique[j + 1]) / 2
                for op in (0,1):
                    pred_label = (feature >= thres)*2-1 if op==1 else (feature<thres)*2-1
                    err = np.sum((pred_label != train_label) * sample_weight)
                    if err < self.err:  # 选取最低错误率
                        self.err = err
                        self.best_fea_index = i
                        self.best_thres = thres
                        self.best_op=op
        return self


    def predict(self, test_data):
        feature = test_data[:, self.best_fea_index]
        return (feature >= self.best_thres)*2-1 if self.best_op==1 else (feature<self.best_thres)*2-1



class Adaboost:
    def __init__(self, num_base=100, base=1):
        self.num_base = num_base
        self.estimators = []
        self.alphas = []
        self.base=base
        self.best_score=0

    def fit(self,x_file,y_file):
        #读取训练文件
        train_raw_data = open(x_file, 'rt')
        train_data = np.loadtxt(train_raw_data, delimiter=",")
        train_raw_data.close()
        #归一化
        train_data_max = np.max(train_data)
        train_data_min = np.min(train_data)
        train_data = (train_data - train_data_min) / (train_data_max - train_data_min)
        #读取标签文件
        train_raw_label = open(y_file)
        train_label = np.loadtxt(train_raw_label, delimiter=',')
        train_raw_label.close()
        #打乱样本顺序
        index=[i for i in range(len(train_label))]
        np.random.shuffle(index)
        train_data=train_data[index]
        train_label=train_label[index]
        index=[i+1 for i in index]

        #十折交叉验证
        sample_num=len(train_label)
        step = int(sample_num / 10)
        l = int(0)
        r = step
        best_estimators=[]
        best_alphas=[]
        best_score=0
        for i in range(1,11):
            temp_train_data=np.append(train_data[0:l,:],train_data[r:sample_num,:],axis=0)
            temp_train_label=np.append(train_label[0:l],train_label[r:sample_num])
            self.crossFit(temp_train_data,temp_train_label)
            temp_score=self.score(train_data[l:r,:],train_label[l:r])
            save_csv=np.array(index[l:r]).reshape((step,1))
            save_csv=np.append(save_csv,self.innerPredict(train_data[l:r,:]).reshape((step,1)),axis=1)
            np.savetxt('data/experiments/base%d_fold%d.csv' % (self.num_base, i), save_csv, fmt='%i', delimiter=',')
            if(best_score<temp_score):
                best_score=temp_score
                best_alphas=self.alphas
                best_estimators=self.estimators
            self.estimators=[]
            self.alphas=[]
            l += step
            r += step
        self.estimators=best_estimators
        self.alphas=best_alphas
        self.best_score=best_score

    def crossFit(self, train_data, train_label):
        sample_weight = np.ones(len(train_data)) / len(train_data)  # 初始化样本权重为 1/N
        temp_train_label=train_label*2-1
        epsilon=1e-5
        for _ in range(self.num_base):
            if(self.base==1):
                weak_learn = DecisionTreeClassifierWithWeight().fit(train_data, temp_train_label, sample_weight)  # 训练弱学习器
            else:
                weak_learn= LogisticRegressionWithWeight().fit(train_data, train_label, sample_weight)
            if weak_learn.err>0.49:    #比随机猜测差则终止
                break
            alpha = 1/2 * np.log((1 - weak_learn.err+epsilon) / (weak_learn.err+epsilon))  # 权重系数
            pred_label = weak_learn.predict(train_data)
            for i in range(len(train_data)):
                if pred_label[i]==temp_train_label[i]:
                    sample_weight[i] *= np.exp(-alpha)  # 更新迭代样本权重
                else:
                    sample_weight[i]*=np.exp(alpha)
            sample_weight /= np.sum(sample_weight)  # 样本权重归一化
            self.estimators.append(weak_learn)
            self.alphas.append(alpha)
        return self

    def predict(self,x_file):
        test_raw_data=open(x_file)
        test_data = np.loadtxt(test_raw_data, delimiter=',')
        test_raw_data.close()
        return self.innerPredict(test_data)

    def innerPredict(self, test_data):
        pred_label = np.empty((len(test_data), len(self.estimators)))
        for i in range(len(self.estimators)):
            pred_label[:, i] = self.estimators[i].predict(test_data)
        pred_label = pred_label * np.array(self.alphas)  # 将预测结果与训练权重乘积作为集成预测结果
        return (np.sum(pred_label, axis=1)>0)  # 以0为阈值，结果映射到0和1

    def score(self,test_data,test_label):
        y_pred = self.innerPredict(test_data)
        return np.mean(y_pred == test_label)



def main():
    base_list = [1, 5, 10, 100]
    best_score=0
    best_clf=None
    for i in base_list:
        clf=Adaboost(base=1 , num_base=i)  #base=0为以逻辑回归作为基学习器，base=1为以决策树作为基学习器，评测请修改base得到两种学习器的训练结果
        x_file="data/data.csv"
        y_file="data/targets.csv"
        clf.fit(x_file,y_file)
        #选取正确率最高的模型
        if best_score<clf.best_score:
            best_clf=clf
            best_score=clf.best_score
    print("训练结束，可以直接使用项目中的evaluate.py来得到十折交叉验证的结果")
    #predict_file为预测数据文件，输入预测文件路径运行
    #predict_file=input('输入测试文件路径:')
    #predict_result=best_clf.predict(predict_file)
    #预测结果保存到data/predict_output.csv
    #np.savetxt('data/predict_output.csv', predict_result, fmt='%i', delimiter=',')

if __name__ == '__main__':
        main()
