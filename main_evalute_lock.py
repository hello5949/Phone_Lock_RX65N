import numpy as np
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import csv
from keras import backend as K
import keras
from keras.models import load_model
import tensorflow as tf
# def getSample(path):
	# label = []
	# input = []
	# with open(path, 'r', encoding='utf-8') as data:
		# read = csv.reader(data)
		# first_skip=True
		# for line in read:
			# if first_skip:
				# first_skip=False
				# continue
			# #one_hot=np.zeros(output_size)
			# #one_hot[int(line[0])]=1
			
			# label.append(int(line[0]))
			# raw = []
			# for i in line[1:]:
				# num=float(i)
				# if num>0:
					# raw.append(num)
				# else:
					# raw.append(0)
			# raw=np.array(raw)
			# raw=raw/np.average(raw)
			# input.append(raw)
	# return np.array(input),np.array(label)

# x,y = getSample("sample_14L_Amptitute_0529.csv")
# # x,y = getSample("sample_mid.csv")
# print(np.shape(x), np.shape(y))


output_size=5 #類別數
input_size = 8192 #輸入Feature大小
ClassSampleNum = 325 #每個類別的樣本數
verifSetNum = 75
TestSetNum = 50
f_min = 150
f_max = 70000
Resolution = 140000 / 16384
def getSample(path):
	label = []
	input = []
	with open(path, 'r', encoding='utf-8') as data:
		read = csv.reader(data)
		first_skip=False
		for line in read:
			if first_skip:
				first_skip=False
				continue
			#one_hot=np.zeros(output_size)
			#one_hot[int(line[0])]=1
			
			label.append(int(line[0]))
			raw = []
			for i in line[0::]:
				num=float(i)
				if num>0:
					raw.append(num)
				else:
					raw.append(0)
			raw=np.array(raw)
			raw=raw/np.average(raw)
			input.append(raw)
	return np.array(input),np.array(label)

x,y = getSample("sample_clear.csv")
print(np.shape(x))

std_x = np.std(x)
mean_x = np.mean(x)
x = (x-mean_x)/std_x

def sampleSplit(x, trainNum, validNum, testNum):
    totleNum = trainNum+ validNum+ testNum
    if totleNum > ClassSampleNum:
        return 0
    x_train = []
    x_valid = []
    x_test = []
    for i in range(output_size):
        np.random.shuffle(x[i*ClassSampleNum:(i+1)*ClassSampleNum])
        x_train.append(x[i*ClassSampleNum :i*ClassSampleNum +trainNum])
        x_valid.append(x[i*ClassSampleNum +trainNum:i*ClassSampleNum +trainNum+ validNum])
        x_test.append(x[i*ClassSampleNum +trainNum+ validNum:i*ClassSampleNum +trainNum+ validNum+ testNum])

    return x_train, x_valid, x_test

x_train, x_valid, x_test = sampleSplit(x,ClassSampleNum-verifSetNum-TestSetNum,verifSetNum,TestSetNum)
x_train = np.reshape(x_train, (output_size*(ClassSampleNum-verifSetNum-TestSetNum),input_size,1))
x_valid = np.reshape(x_valid, (output_size*verifSetNum,input_size,1))
x_test = np.reshape(x_test, (output_size*TestSetNum,input_size,1))
x_train = np.array(x_train)
x_valid = np.array(x_valid)
x_test = np.array(x_test)
print(np.shape(x_train))
print(np.shape(x_valid))
print(np.shape(x_test))


def Reorganize(x):
    x = np.reshape(x, (len(x), 8192))
    x_mean_col = np.mean(x , axis = 0)
    print(x_mean_col)
    x_mean = np.mean(x_mean_col[int((150/Resolution))::])*0.4
    print(x_mean)
    f = -Resolution
    flag = 0
    freq = []
    Idx = []
    #Select region
    print("Select region")
    for i in range(input_size):
        f += Resolution
        if f > f_min and x_mean_col[i] > x_mean:
            if flag == 0:
                flag = 1
                freq.append([f])
                Idx.append([i])
        if f > f_min and x_mean_col[i] < x_mean:
            if flag == 1:
                flag = 0
                freq[len(freq)-1].append(f)
                Idx[len(Idx)-1].append(i)
    
    #merge
    print("merge")
    j=0
    freq = np.array(freq)
    Idx = np.array(Idx)
    for i in range(len(freq)):
        if i == 0:
            continue
        if freq[j+1][0] - freq[j][1] < 300:
            freq[j][1] = freq[j+1][0]
            Idx[j][1] = Idx[j+1][0]
            freq = np.delete(freq, j+1, 0)
            Idx = np.delete(Idx, j+1, 0)
        else:
            j += 1
            
    #Reorganize
    print("Reorganize")
    new_x = []
    for i in range(output_size*ClassSampleNum):
        new_x.append([])
        for idx in Idx:
            new_x[i].extend(x[i][idx[0]:idx[1]])
    
    # plt.plot(new_x[0][0:2369])
    # plt.title("Feature Map")
    # plt.xlabel("Feature point")
    # plt.ylabel("Amplitude")
    # plt.show()
    
    new_x = np.reshape(new_x,(len(new_x), len(new_x[0])))
    print(np.shape(new_x))
    print("Freq. region : ")
    print(freq)
    print(Idx)
    
    # x_label = np.reshape(freq,(len(freq)*2))
    # y_label = np.ones(len(x_label))
    # plt.plot(np.arange(8191)*(Resolution)+Resolution, x_mean_col)
    # plt.bar(x_label,10,100, color='r')
    # plt.plot([0,70000], [x_mean, x_mean])
    # plt.xlabel("Frequency")
    # plt.ylabel("Amplitude")
    # plt.legend(["Signal", "Threshold", "Select_Region"])
    # plt.show()
    return new_x, len(new_x[0])
    
# x, input_size = Reorganize(x)

def Greedy(evaluate_result, Model_num, sample_num, G_index = None, Threshold = None):
    TP=0
    TN=0
    FP=0
    FN=0

    if Threshold == None:
        sort_lose = np.sort(evaluate_result)
        Threshold = sort_lose[Model_num][G_index]
    
    for i,lose_ in enumerate(evaluate_result[Model_num]):
        if lose_ > Threshold: ##樣本lose大於當前閥值，即判定為不合格
            if (i%len(evaluate_result[Model_num]) >= Model_num*sample_num) and (i%len(evaluate_result[Model_num]) < (Model_num+1)*sample_num): ##實際為合格
                FP += 1
            else:##實際為不合格
                TP += 1
        else: ##樣本lose小於當前閥值，即判定為合格
            if (i%len(evaluate_result[Model_num]) >= Model_num*sample_num) and (i%len(evaluate_result[Model_num]) < (Model_num+1)*sample_num): ##實際為合格
                TN += 1
            else: #實際為不合格
                FN += 1
    TPR = TP/(FN+TP)
    FPR = FP/(FP+TN)
    PRE = TP/(TP+FP)
    ACC = (TP+TN)/(TP+TN+FP+FN)
    
    return TPR, FPR, PRE, ACC, TP, TN, FP, FN

pred_Data_test = []
pred_Data_train = []
for i in range(output_size*ClassSampleNum):
    if (i%ClassSampleNum >= ClassSampleNum-TestSetNum):
        pred_Data_test.append([x[i]])
    else:
        pred_Data_train.append([x[i]])
pred_Data_test = np.array(pred_Data_test)
pred_Data_train = np.array(pred_Data_train)
print(np.shape(pred_Data_test))

##讀取所有模型
class_model = []
for i in range(output_size):
    class_model.append(load_model('./AE_Model/model_'+repr(i)+'/model_'+repr(i)+'.h5'))
    # class_model.append(tf.keras.models.load_model("model_"+repr(i)+".h5"))
    
##跑lose
evaluate_result_test = []
evaluate_result_valid = []
evaluate_result_train = []
for i in range(output_size):
    cc = []
    for j in range(output_size*(ClassSampleNum-verifSetNum-TestSetNum)):
        cc.append(class_model[i].evaluate(x_train[j].T,x_train[j].T)) ## test set
    evaluate_result_train.append(cc)
    cc = []
    for j in range(output_size*TestSetNum):
        cc.append(class_model[i].evaluate(x_test[j].T,x_test[j].T)) ## test set
    evaluate_result_test.append(cc)
    cc = []
    for j in range(verifSetNum*output_size):
        cc.append(class_model[i].evaluate(x_valid[j].T,x_valid[j].T))  ## train set
    evaluate_result_valid.append(cc)

#Greedy AE
sample_num = verifSetNum  #test sample num per class
TPR_list = []
FPR_list = []
PRE_list = []
ACC_list = []
TP_list = []
TN_list = []
FP_list = []
FN_list = []
Threshold_list = []
sort_lose = np.sort(evaluate_result_valid)
print(len(evaluate_result_valid[0]), np.log2(len(evaluate_result_valid[0])))
for Model_num in range(output_size):
    G_index = int(len(sort_lose[0])/2)
    min_index = 0
    max_index = len(evaluate_result_valid[Model_num])
    MAX_ACC = 0
    for j in range(int(np.log2(len(evaluate_result_valid[Model_num])))+20):

        TPR, FPR, PRE, ACC, TP, TN, FP, FN = Greedy(evaluate_result_valid, Model_num, sample_num, G_index)
        
        if MAX_ACC == 0:
            MAX_ACC = ACC
        if ACC >= MAX_ACC:
            MAX_ACC = ACC
            index_bias = 0
            while ACC == MAX_ACC:
                index_bias = index_bias + 1
                TPR, FPR, PRE, ACC, TP, TN, FP, FN = Greedy(evaluate_result_valid, Model_num, sample_num, G_index+index_bias)
                # if((G_index+index_bias) >= max_index and ACC < MAX_ACC):
                    # max_index = G_index
                    # G_index = int((G_index+min_index)/2)
                    # break
                if ACC > MAX_ACC:
                    MAX_ACC = ACC
                    G_index = G_index+index_bias
                    break
                    # min_index = G_index
                    # G_index = int((G_index+max_index)/2)
                elif ACC < MAX_ACC:
                    max_index = G_index
                    G_index = int((G_index+min_index)/2)
        elif ACC < MAX_ACC:
            min_index = G_index
            G_index = int((G_index+max_index)/2)
        TPR, FPR, PRE, ACC, TP, TN, FP, FN = Greedy(evaluate_result_valid, Model_num, sample_num, G_index+index_bias)
        # print(ACC, MAX_ACC, min_index, G_index, max_index)
        print(Model_num, ACC, MAX_ACC, min_index, G_index, max_index, "砍人 : ", TP, TN, FP, FN)
                
    Threshold_list.append(sort_lose[Model_num][G_index+1])
    TPR_list.append(TPR)
    FPR_list.append(FPR)
    PRE_list.append(PRE)
    ACC_list.append(ACC)
    TP_list.append(TP)
    TN_list.append(TN)
    FP_list.append(FP)
    FN_list.append(FN)
print("TPR : ", TPR_list)
print("FPR : ", FPR_list)
print("PRE : ", PRE_list)
print("ACC : ", ACC_list)
print("Threshold : ", Threshold_list)

for i in range(3):
    TPR_list.append("")
    FPR_list.append("")
    PRE_list.append("")
    ACC_list.append("")
    TP_list.append("")
    TN_list.append("")
    FP_list.append("")
    FN_list.append("")
    
for Model_num in range(output_size):
    TPR, FPR, PRE, ACC, TP, TN, FP, FN = Greedy(evaluate_result_test, Model_num, TestSetNum, Threshold = Threshold_list[Model_num])
    TPR_list.append(TPR)
    FPR_list.append(FPR)
    PRE_list.append(PRE)
    ACC_list.append(ACC)
    TP_list.append(TP)
    TN_list.append(TN)
    FP_list.append(FP)
    FN_list.append(FN)

for i in range(3):
    TPR_list.append("")
    FPR_list.append("")
    PRE_list.append("")
    ACC_list.append("")
    TP_list.append("")
    TN_list.append("")
    FP_list.append("")
    FN_list.append("")
    
print("TPR : ", TPR_list)
print("FPR : ", FPR_list)
print("PRE : ", PRE_list)
print("ACC : ", ACC_list)
print("TP : ", TP_list)
print("TN : ", TN_list)
print("FP : ", FP_list)
print("FN : ", FN_list)
print(np.shape(evaluate_result_valid), np.shape(evaluate_result_test))
with open('Evalute_Result.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['TP', 'TN','FP','FN','TPR','FPR','PRE','ACC'])
    for i in range(len(TPR_list)):
        writer.writerow([TP_list[i],  TN_list[i], FP_list[i], FN_list[i], TPR_list[i], FPR_list[i], PRE_list[i], ACC_list[i]])

    writer.writerow(['Model_1_Train', 'Model_2_Train','Model_3_Train','Model_4_Train','Model_5_Train'])
    writer.writerows(evaluate_result_valid)
    writer.writerow(['Model_1_Test', 'Model_2_Test','Model_3_Test','Model_4_Test','Model_5_Test'])
    writer.writerows(evaluate_result_test)
    

model_name = ['Model_I-7p','Model_I-8p','Model_Sam-A7','LG','Model_I-XR']
print(np.shape(model_name))
x_threshold = np.arange(output_size*verifSetNum)

for i in range(output_size):
    y_threshold = np.ones(output_size*verifSetNum)*Threshold_list[i]
    plt.plot(evaluate_result_valid[i])
    plt.plot(x_threshold,y_threshold)
    plt.title("AE model_"+repr(model_name[i][:]))
    plt.grid(color='gray', linestyle='-', linewidth=0.5)
    plt.xlabel('    I-7p               I-8p          Sam-A7           LG           I-XR')
    plt.ylabel('loss')
    # plt.legend(model_name)
    new_ticks = np.linspace(0, output_size*verifSetNum, output_size+1)
    plt.xticks(new_ticks)
    plt.show()
    
x_threshold = np.arange(output_size*TestSetNum)
for i in range(output_size):
    y_threshold = np.ones(output_size*TestSetNum)*Threshold_list[i]
    plt.plot(evaluate_result_test[i])
    plt.plot(x_threshold,y_threshold)
    plt.title("AE model_"+repr(model_name[i][:]))
    plt.grid(color='gray', linestyle='-', linewidth=0.5)
    plt.xlabel('    I-7p               I-8p          Sam-A7           LG           I-XR')
    plt.ylabel('loss')
    # plt.legend(model_name)
    new_ticks = np.linspace(0, output_size*TestSetNum, output_size+1)
    plt.xticks(new_ticks)
    plt.show()
    
