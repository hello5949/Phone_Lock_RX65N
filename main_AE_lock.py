import numpy as np
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Input, BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import csv
from keras import backend as K
from keras.callbacks import EarlyStopping
from hyperopt import hp, fmin, rand, hp, STATUS_OK, Trials,tpe
from hyperopt.mongoexp import MongoTrials
# from tensorflow.keras import backend as K
# from tensorflow.python.keras import backend as K

output_size=5 #參數大小
input_size = 8192 #輸入Feature大小
ClassSampleNum = 300 #每個類別的樣本數
verifSetNum = 75
TestSetNum = 75
isTraining = True
Training_Data = "Train2.csv"
Testing_Data = "Test.csv"

space = {
	'learning_rate': 0.014818872,
	# 'batch_size' : hp.randint('batch_size', 1024),
	'activation' : 'relu',
	'units' : 32,
	'layers_count' : 0,
	'loss_func' : 'mse',
}

def getSample(path):
	label = []
	input = []
	with open(path, 'r', encoding='utf-8') as data:
		read = csv.reader(data)
		first_skip=True
		for line in read:
			if first_skip:
				first_skip=False
				continue
			#one_hot=np.zeros(output_size)
			#one_hot[int(line[0])]=1
			
			# label.append(int(line[0]))
			raw = []
			for i in line[0::]:
				num=float(i)
				if num>0:
					raw.append(np.log(num))
				else:
					raw.append(0)
			raw=np.array(raw)
			# raw=raw/np.average(raw)
			input.append(raw)
	return np.array(input)

x_train = getSample(Training_Data)
x_test = getSample(Testing_Data)
x_varif = []
dd = []
for i in range(output_size*(verifSetNum+TestSetNum)):
    if i%(verifSetNum + TestSetNum) < verifSetNum:
        x_varif.append(x_test[i])
    elif i%(verifSetNum + TestSetNum) < (verifSetNum + TestSetNum):
        dd.append(x_test[i])
x_test = dd
x_train = np.array(x_train)
x_varif = np.array(x_varif)
x_test = np.array(x_test)
print(np.shape(x_train), np.shape(x_varif), np.shape(x_test))

def LayerCreate(layerNum, maxNode, Activation, model, room=None):
    node = maxNode
    nodeList = []
    if room == None:
        room = 2
    # encoder
    model.add(Dense(maxNode, activation=Activation, input_shape = (input_size,)))
    print(Activation, input_size)
    nodeList.append(maxNode)
    if layerNum == 0 or maxNode == 1:
        model.add(Dense(input_size))
        return 0

    for i in range(layerNum):
        node = int(node/room)
        nodeList.append(node)
        model.add(Dense(node))
        if node <=1:
            break
    # decoder
    for i in range(len(nodeList)):
        model.add(Dense(nodeList[len(nodeList)-i-1]))
    model.add(Dense(input_size))


def Greedy(Test_Eval_loss, Model_num, sample_num, G_index = None, Threshold = None):
    TP=0
    TN=0
    FP=0
    FN=0

    if Threshold == None:
        sort_lose = np.sort(Test_Eval_loss)
        Threshold = sort_lose[Model_num][G_index]
    
    for i,lose_ in enumerate(Test_Eval_loss[Model_num]):
        if lose_ > Threshold: ##樣本lose大於當前閥值，即判定為不合格
            if (i%len(Test_Eval_loss[Model_num]) >= Model_num*sample_num) and (i%len(Test_Eval_loss[Model_num]) < (Model_num+1)*sample_num): ##實際為合格
                FP += 1
            else:##實際為不合格
                TP += 1
        else: ##樣本lose小於當前閥值，即判定為合格
            if (i%len(Test_Eval_loss[Model_num]) >= Model_num*sample_num) and (i%len(Test_Eval_loss[Model_num]) < (Model_num+1)*sample_num): ##實際為合格
                TN += 1
            else: #實際為不合格
                FN += 1
    TPR = TP/(FN+TP)
    FPR = FP/(FP+TN)
    PRE = TP/(TP+FP)
    ACC = (TP+TN)/(TP+TN+FP+FN)
    
    return TPR, FPR, PRE, ACC, TP, TN, FP, FN

# x, input_size = Reorganize(x)
# std_train = np.std(x_train)
# mean_train = np.mean(x_train)
# x_train = (x_train-mean_train)/std_train
# print(std_train, mean_train)

# std_valid = np.std(x_varif)
# mean_valid = np.mean(x_varif)
# x_varif = (x_varif-mean_valid)/std_valid
# print(std_valid, mean_valid)

# std_test = np.std(x_test)
# mean_test = np.mean(x_test)
# x_test = (x_test-mean_test)/std_test
# print(std_test, mean_test)


# in order to plot in a 2D figure
encoding_dim = 2
step = 0
loss_set = []
ACC_set = []
layer_set = []
zoom_set = []
lr_set = []
node_set = []
layer_set = []
activation_set = []
loss_func_set = []
evaluate_history = []


def AutoEncoder(params):
    LR = params['learning_rate'] 
    Node = int(params['units'])
    Activation = params['activation'] 
    Loss_func = params['loss_func'] 
    Layer = params['layers_count']

    batch_size = 375
    print("params:", LR, Node, Activation, Loss_func, Layer, batch_size)
    if isTraining:
        for j in range(output_size):
            print("Node:", Node, "LR:", LR ,"Activation:", Activation, "Loss_func:", Loss_func)
            K.clear_session()
            model = Sequential()
            # model.add(BatchNormalization(axis=-1, epsilon=0.001, center=True, input_shape = (input_size,)))   
            LayerCreate(Layer, Node, Activation, model, 2)
            print(model.summary())
            adam = Adam(lr=LR)
            model.compile(optimizer='adam', loss=Loss_func)
            ES_Acc = EarlyStopping(monitor='val_loss',min_delta=0, mode='min', verbose=1, patience=200)
            history = model.fit(x_train[j*ClassSampleNum:((j+1)*ClassSampleNum)-1,:], x_train[j*ClassSampleNum:((j+1)*ClassSampleNum)-1,:], 
            epochs=3000, batch_size=int(batch_size), shuffle=True, callbacks=([ES_Acc]), 
            validation_data=(x_varif[(j*TestSetNum):(j+1)*TestSetNum-1], x_varif[(j*TestSetNum):(j+1)*TestSetNum-1])) 
            model.save('./AE_Model/model_'+repr(j)+'/model_'+repr(j)+'.h5')

    

# 分配驗證與測試資料  (*評估輸入shape問題，重新分配)
    Eval_Test_Data = []
    Eval_Varif_Data = []
    Eval_Train_Data = []
    for i in range(output_size*TestSetNum):
        Eval_Test_Data.append([x_test[i]])
    Eval_Test_Data = np.array(Eval_Test_Data)
    for i in range(output_size*verifSetNum):
        Eval_Varif_Data.append([x_varif[i]])
    Eval_Varif_Data = np.array(Eval_Varif_Data)
    for i in range(output_size*ClassSampleNum):
        Eval_Train_Data.append([x_train[i]])
    Eval_Train_Data = np.array(Eval_Train_Data)
    
    print(np.shape(Eval_Test_Data))
    #讀取所有模型
    class_model = []
    for i in range(output_size):
        class_model.append(load_model('./AE_Model/model_'+repr(i)+'/model_'+repr(i)+'.h5'))

    #跑LOSS
    Test_Eval_loss = []
    Varif_Eval_loss = []
    Train_Eval_loss = []
    for i in range(output_size):
        cc = []
        for j in range(output_size*TestSetNum):
            cc.append(class_model[i].evaluate(Eval_Test_Data[j],Eval_Test_Data[j]))
        Test_Eval_loss.append(cc)
        cc = []
        for j in range(output_size*verifSetNum):
            cc.append(class_model[i].evaluate(Eval_Varif_Data[j],Eval_Varif_Data[j]))
        Varif_Eval_loss.append(cc)
        # c = []
        # for j in range(output_size*verifSetNum):
        #     cc.append(class_model[i].evaluate(Eval_Train_Data[j],Eval_Train_Data[j]))
        # Train_Eval_loss.append(cc)
        
    ##save the loss of evaluate to csv
    evaluate_history.extend(Varif_Eval_loss)
    evaluate_history.extend(Test_Eval_loss)
    with open('GS_AE_0807_evaluateResult.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Set','Phone_1', 'Phone_2', 'Phone_3','Phone_4','Phone_4'])
        writer.writerows(evaluate_history)

    #畫圖&存圖
    for i in range(output_size):
        plt.plot(Test_Eval_loss[i])
    plt.grid(color='gray', linestyle='-', linewidth=0.5)
    plt.xlabel('-   I-7p           I-8p          Sam-A7         Sam-A7-2        Sony  -')
    plt.ylabel('loss')
    plt.legend(['Model_I-7p','Model_I-8p','Model_Sam-A7','Model_Sam-A7-2','Model_Sony','Model_Sony','Model_L6','Model_L7','Model_L8','Model_L9','Model_L10','Model_L11','Model_L12','Model_L13','Model_L14'])
    plt.xticks(np.arange(7)*75, ['1', '75  1    ', '75  1    ', '75  1    ', '75  1    ', '75  '])
    plt.title("Layer:" +repr(Layer)+ "Node:" +repr(Node)+ "LR:"+ repr(LR)+ "Acti.:"+ repr(Activation)+ "Loss_func:"+ repr(Loss_func))
    plt.show()

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
    sort_lose = np.sort(Varif_Eval_loss)
    print(len(Varif_Eval_loss[0]), np.log2(len(Varif_Eval_loss[0])))
    for Model_num in range(output_size):
        G_index = int(len(sort_lose[0])/2)
        min_index = 0
        max_index = len(Varif_Eval_loss[Model_num])
        MAX_ACC = 0
        for j in range(int(np.log2(len(Varif_Eval_loss[Model_num])))+20):

            TPR, FPR, PRE, ACC, TP, TN, FP, FN = Greedy(Varif_Eval_loss, Model_num, sample_num, G_index)
            
            if MAX_ACC == 0:
                MAX_ACC = ACC
            if ACC >= MAX_ACC:
                MAX_ACC = ACC
                index_bias = 0
                while ACC == MAX_ACC:
                    index_bias = index_bias + 1
                    TPR, FPR, PRE, ACC, TP, TN, FP, FN = Greedy(Varif_Eval_loss, Model_num, sample_num, G_index+index_bias)
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
            TPR, FPR, PRE, ACC, TP, TN, FP, FN = Greedy(Varif_Eval_loss, Model_num, sample_num, G_index+index_bias)
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

    TPR_list = []
    FPR_list = []
    PRE_list = []
    ACC_list = []
    TP_list = []
    TN_list = []
    FP_list = []
    FN_list = []
        
    for Model_num in range(output_size):
        TPR, FPR, PRE, ACC, TP, TN, FP, FN = Greedy(Varif_Eval_loss, Model_num, TestSetNum, Threshold = Threshold_list[Model_num])
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
    print("Verification Result:")
    print("TPR : ", TPR_list)
    print("FPR : ", FPR_list)
    print("PRE : ", PRE_list)
    print("ACC : ", ACC_list)
    print("TP : ", TP_list)
    print("TN : ", TN_list)
    print("FP : ", FP_list)
    print("FN : ", FN_list)

    for Model_num in range(output_size):
        TPR, FPR, PRE, ACC, TP, TN, FP, FN = Greedy(Test_Eval_loss, Model_num, TestSetNum, Threshold = Threshold_list[Model_num])
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
    
    print("Test Result")
    print("TPR : ", TPR_list)
    print("FPR : ", FPR_list)
    print("PRE : ", PRE_list)
    print("ACC : ", ACC_list)
    print("TP : ", TP_list)
    print("TN : ", TN_list)
    print("FP : ", FP_list)
    print("FN : ", FN_list)


    with open('Evalute_Result.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['TP', 'TN','FP','FN','TPR','FPR','PRE','ACC'])
        for i in range(len(TPR_list)):
            writer.writerow([TP_list[i],  TN_list[i], FP_list[i], FN_list[i], TPR_list[i], FPR_list[i], PRE_list[i], ACC_list[i]])

    model_name = ['Model_I-7p','Model_I-8p','Model_Sam-A7','LG','Model_I-XR']
    x_threshold = np.arange(output_size*verifSetNum)

    for i in range(output_size):
        y_threshold = np.ones(output_size*verifSetNum)*Threshold_list[i]
        plt.plot(Varif_Eval_loss[i])
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
        plt.plot(Test_Eval_loss[i])
        plt.plot(x_threshold,y_threshold)
        plt.title("AE model_"+repr(model_name[i][:]))
        plt.grid(color='gray', linestyle='-', linewidth=0.5)
        plt.xlabel('    I-7p               I-8p          Sam-A7           LG           I-XR')
        plt.ylabel('loss')
        # plt.legend(model_name)
        new_ticks = np.linspace(0, output_size*TestSetNum, output_size+1)
        plt.xticks(new_ticks)
        plt.show()
    
    global step
    step = step + 1
    return 0

AutoEncoder(space)