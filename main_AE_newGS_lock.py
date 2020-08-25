import numpy as np
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Input, BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import csv
from keras import backend as K
import keras
from keras.callbacks import EarlyStopping
# from tensorflow.keras import backend as K
# from tensorflow.python.keras import backend as K

output_size=5 #參數大小
input_size = 8192 #輸入Feature大小
ClassSampleNum = 250 #每個類別的樣本數
TestSetNum = 100
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
print(x.shape)
print(y.shape)  


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


def LayerCreate(layerNum, maxNode, Activation, room=None):
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

# x, input_size = Reorganize(x)
print(np.shape(x), np.shape(y))
std_x = np.std(x)
mean_x = np.mean(x)
x = (x-mean_x)/std_x
print(std_x, mean_x, x)

# in order to plot in a 2D figure
encoding_dim = 2
step = 0
loss_set = []
layer_set = []
zoom_set = []
lr_set = []
node_set = []
layer_set = []
activation_set = []
loss_func_set = []
evaluate_history = []

# GS parameter
Nodes = [10,9,8,7,6,5,4,3,2,1]
Layers = [2,1,0]
LRs = [0.001,0.01]
Activations = ['relu','tanh','sigmoid']
Loss_funcs = ['mae']
# GS parameter:layer zoom lr activation loss_func
for Node in Nodes:
    for Layer in Layers:
        for LR in LRs:
            for Activation in Activations:
                for Loss_func in Loss_funcs:
                    step = step + 1
                    for j in range(output_size):
                        print("Node:", Node, "LR:", LR ,"Activation:", Activation, "Loss_func:", Loss_func)
                        K.clear_session()
                        model = Sequential()
                        # model.add(BatchNormalization(axis=-1, epsilon=0.001, center=True, input_shape = (input_size,)))   
                        LayerCreate(Layer, Node, Activation, 2)
                        print(model.summary())
                        adam = Adam(lr=LR)
                        model.compile(optimizer='adam', loss=Loss_func)
                        ES_Acc = EarlyStopping(monitor='val_loss',min_delta=0, mode='min', verbose=1, patience=50)
                        history = model.fit(x[j*ClassSampleNum:((j+1)*ClassSampleNum)-TestSetNum-1,:], x[j*ClassSampleNum:((j+1)*ClassSampleNum)-TestSetNum-1,:], 
                        epochs=600, batch_size=600, shuffle=True, callbacks=([ES_Acc]), 
                        validation_data=(x[(j*ClassSampleNum)+(ClassSampleNum-TestSetNum):(j+1)*ClassSampleNum-1], x[(j*ClassSampleNum)+(ClassSampleNum-TestSetNum):(j+1)*ClassSampleNum-1]))
                        model.save('./AE_Model/model_'+repr(j)+'/model_'+repr(j)+'.h5')
                        
                    loss_set.append(min(history.history['loss']))
                    node_set.append(Node)
                    layer_set.append(Layer)
                    lr_set.append(LR)
                    activation_set.append(Activation)
                    loss_func_set.append(Loss_func)
                    with open('GS_AE_0807_5p_Nor_STD_layer_clear.csv', 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(['Set', 'loss', 'Layer','Node','LR','Activation','loss_Func'])
                        for i in range(len(loss_set)):
                            writer.writerow([i+1,  loss_set[i], layer_set[i], node_set[i], lr_set[i], activation_set[i], loss_func_set[i]])
                    
                    pred_Data = []
                    for i in range(output_size*ClassSampleNum):
                        if (i%ClassSampleNum >= (ClassSampleNum-TestSetNum)):
                            pred_Data.append([x[i]])
                    pred_Data = np.array(pred_Data)
                    print(np.shape(pred_Data))
                    class_model = []
                    for i in range(output_size):
                        class_model.append(load_model('./AE_Model/model_'+repr(i)+'/model_'+repr(i)+'.h5'))
                        # class_model.append(tf.keras.models.load_model("model_"+repr(i)+".h5"))
                    evaluate_result = []
                    for i in range(output_size):
                        cc = []
                        for j in range(output_size*TestSetNum):
                            cc.append(class_model[i].evaluate(pred_Data[j],pred_Data[j]))
                        evaluate_result.append(cc)
                        
                    ##save the loss of evaluate to csv
                    evaluate_history.extend(evaluate_result)
                    with open('GS_AE_0807_evaluateResult.csv', 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(['Set','Phone_1', 'Phone_2', 'Phone_3','Phone_4','Phone_4'])
                        writer.writerows(evaluate_history)

                    #畫圖&存圖
                    for i in range(output_size):
                        plt.plot(evaluate_result[i])
                    plt.grid(color='gray', linestyle='-', linewidth=0.5)
                    plt.xlabel('  I-7p                I-8p            Sam-A7          LG              I-8p')
                    plt.ylabel('loss')
                    plt.legend(['Model_I-7p','Model_I-8p','Model_Sam-A7','Model_LG','Model_I-XR','Model_L6','Model_L7','Model_L8','Model_L9','Model_L10','Model_L11','Model_L12','Model_L13','Model_L14'])
                    new_ticks = np.linspace(0, output_size*TestSetNum, output_size+1)
                    plt.xticks(new_ticks)
                    plt.title("Layer:" +repr(Layer)+ "Node:" +repr(Node)+ "LR:"+ repr(LR)+ "Acti.:"+ repr(Activation)+ "Loss_func:"+ repr(Loss_func))
                    filename1 = '.\GS_Result/Set%03d.png' % (step)
                    plt.savefig(filename1)
                    plt.close()
                