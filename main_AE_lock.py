import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Input, BatchNormalization
import matplotlib.pyplot as plt
import csv
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import EarlyStopping
import keras
import time
from tensorflow.keras import layers
from tensorflow.keras import activations
from keras.models import load_model
import tensorflow as tf

output_size=5 #參數大小
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
print(x.shape)
print(y.shape)  

std_x = np.std(x)
mean_x = np.mean(x)
x = (x-mean_x)/std_x
print(std_x, mean_x)

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
x_train = np.array(x_train)
x_valid = np.array(x_valid)
x_test = np.array(x_test)
    

def Reorganize(x):
    x = np.reshape(x, (len(x), 8191))
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
loss_set = []
# train parameter
#Layer Node LR activation lossFunc
parameter_set = [[2,10,0.01,'sigmoid','mae'],[1,10,0.01,'relu','mae'],[1,7,0.01,'relu','mae'],[0,10,0.01,'relu','mae'],[2,9,0.01,'tanh','mae'],[2,8,0.01,'relu','mae'],[1,8,0.001,'relu','mae'],[0,4,0.01,'relu','mae']]
epoch = 5000
step = 0
for hyperparameter in parameter_set:
    step = step + 1
    for i in range(output_size):
        keras.backend.clear_session()
        model = Sequential()
        # model.add(BatchNormalization(axis=-1, epsilon=0.001, center=True, input_shape = (input_size,)))
        # model.add(Dense(int(hyperparameter[0]), activation=hyperparameter[2], input_shape = (input_size,)))
        # model.add(Dense(input_size))
        LayerCreate(hyperparameter[0], hyperparameter[1], hyperparameter[3], 2)
        print(model.summary())
        adam = Adam(lr=float(hyperparameter[2]))
        model.compile(optimizer='adam', loss=hyperparameter[4])
        ES_Acc = EarlyStopping(monitor='val_loss',min_delta=0, mode='min', verbose=1, patience=25)
        history = model.fit(x_train[i], x_train[i],
         epochs=epoch, batch_size=256, shuffle=True, callbacks=([ES_Acc]),
         validation_data=(x_valid[i], x_valid[i]))
        model.save('./AE_Model/model_'+repr(i)+'/model_'+repr(i)+'.h5')
        
        loss_set.append(history.history['loss'])
    with open('./Train_Result/train_result.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Set', 'loss_0', 'loss_1', 'loss_2', 'loss_3', 'loss_4', 'loss_5', 'loss_6', 'loss_7', 'std = ', std_x, 'mean = ', mean_x])
        writer.writerows(loss_set)
        # for i in range(len(loss_set[0])):
        #     writer.writerow([i+1, loss_set[0][i], loss_set[1][i], loss_set[2][i], loss_set[3][i], loss_set[4][i]])

    ############################evalute############################

    pred_Data = []
    for i in range(output_size*ClassSampleNum):
        if (i%ClassSampleNum >= ClassSampleNum-TestSetNum):
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

    for i in range(output_size):
        plt.plot(evaluate_result[i])
    plt.title("Node:" +repr(hyperparameter[0])+ "LR:"+ repr(hyperparameter[1])+ "Activation:"+ repr(hyperparameter[2])+ "Loss_func:"+ repr(hyperparameter[3]))
    plt.grid(color='gray', linestyle='-', linewidth=0.5)
    plt.xlabel('  I-7p                I-8p            Sam-A7          LG              I-8p')
    plt.ylabel('loss')
    plt.legend(['Model_I-7p','Model_I-8p','Model_Sam-A7','Model_LG','Model_I-XR'])
    new_ticks = np.linspace(0, output_size*TestSetNum, output_size+1)
    plt.xticks(new_ticks)
    filename1 = './Train_Result/Set%03d.png' % (step)
    plt.savefig(filename1)
    plt.close()
    plt.show()
    
    
# # in order to plot in a 2D figure
# encoding_dim = 2

# # this is our input placeholder
# input_img = Input(shape=(400,))

# # encoder layers
# encoded = keras.layers.BatchNormalization(axis=1, epsilon=0.001, center=True)(input_img)
# encoded = Dense(128, activation='relu')(encoded)
# encoded = Dense(64, activation='relu')(encoded)
# encoded = Dense(32, activation='relu')(encoded)
# encoded = keras.layers.BatchNormalization(axis=1, epsilon=0.001, center=True)(encoded)
# encoder_output = Dense(encoding_dim)(encoded)

# # decoder layers
# decoded = Dense(32, activation='relu')(encoder_output)
# decoded = Dense(64, activation='relu')(decoded)
# decoded = Dense(128, activation='relu')(decoded)
# decoded = Dense(400)(decoded)

# # construct the autoencoder model
# autoencoder = Model(input=input_img, output=decoded)

# # construct the encoder model for plotting
# encoder = Model(input=input_img, output=encoder_output)

# # compile autoencoder
# autoencoder.compile(optimizer='adam', loss='mse')

# # training
# autoencoder.fit(x, x,
                # epochs=1000,
                # batch_size=256,
                # shuffle=True)

# # plotting
# encoded_imgs = encoder.predict(x)
# plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y)
# plt.colorbar()
# plt.show()

############################evalute############################

pred_Data = []
for i in range(output_size*ClassSampleNum):
    if (i%ClassSampleNum >= ClassSampleNum-TestSetNum):
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

for i in range(output_size):
    plt.plot(evaluate_result[i])
plt.title("AE model_"+repr(i+1))
plt.grid(color='gray', linestyle='-', linewidth=0.5)
plt.xlabel('Sample')
plt.ylabel('loss')
plt.legend(['Model_L1','Model_L2','Model_L3','Model_L4','Model_L5','Model_L6','Model_L7','Model_L8','Model_L9','Model_L10','Model_L11','Model_L12','Model_L13','Model_L14'])
new_ticks = np.linspace(0, output_size*TestSetNum, output_size+1)
plt.xticks(new_ticks)
filename1 = '.\GS_Result/Set%03d.png' % (step)
plt.savefig(filename1)
plt.close()
plt.show()