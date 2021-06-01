from hashlib import new
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
import copy

#전파 -> 입력에서 출력까지 output을 계속 만듬
#역전파 -> 출력에서 입력까지 가중치를 업데이트함. gradient descent를 이용.

def read_file(file) -> list:
    train_list = [ list(map(float, line.rstrip('\n').split(' '))) for line in file.readlines()]
    
    return train_list


def sigmoid(a, alpha = 1):
    ans = 0.0
    try:
        ans = 1/(1+math.exp(-alpha*a))
    except OverflowError:
        ans = 0.0
    return ans


class layer:
    def __init__(self, prev_neuron_num, neuron_num) -> None:
        self.neuron_num = neuron_num
        self.neurons = [[random.random() for weights_num in range(prev_neuron_num)] for num in range(neuron_num)]
        self.prev_neuron_num = prev_neuron_num # 뉴런당 가중치의 갯수임
        self.outputs = []

    def make_output(self, prev_output: list, alpha): # used in forward pass
        self.outputs = []
        for neuron in self.neurons:
            a = 0.0
            for i in range(len(neuron)):
                a += neuron[i] * prev_output[i]
            a = sigmoid(a, alpha)
            self.outputs.append(a)

        return self.outputs


class MLP:
    def __init__(self, input, result_list, alpha, learning_rate, layer_num, layer_neuron_num: list, epoch = 1000000) -> None:
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.layer_num = layer_num
        self.layers = []
        self.layers.append(layer(len(input[0]),layer_neuron_num[0]))
        for i in range(1, layer_num):
            self.layers.append(layer(layer_neuron_num[i-1],layer_neuron_num[i]))
        self.input = input
        self.result_list = result_list
        self.compute(epoch)

    def forwardpass(self, input):#
        self.layers[0].make_output(input, self.alpha)# input에 맞춰 생성

        for i in range(1, self.layer_num):
            self.layers[i].make_output(self.layers[i-1].outputs, self.alpha)
        
    def backpropagation(self, index):
        # 마지막 층 계산하고, 은닉층을 처리한다.
        next = self.layers[-1]
        prev = self.layers[-2]
        prev_layer_Ps = []

        for i in range(next.neuron_num): # 마지막 층
            Pm = (next.outputs[i] - self.result_list[index]) * (next.outputs[i] * (1 - next.outputs[i]))
            prev_layer_Ps.append(Pm)
            for w in range(len(next.neurons[i])): # w is individual weight of neurons[i],                
                next.neurons[i][w] -= (self.learning_rate * Pm * prev.outputs[w])
        
        for layer in range(self.layer_num-2, -1, -1): # 은닉층
            next = self.layers[layer+1]
            now = self.layers[layer]
            prev
            new_Ps = []
            if layer != 0:
                prev = self.layers[layer-1]

            for i in range(now.neuron_num):
                P_m = 0.0

                for m in range(next.neuron_num):
                    P_m += prev_layer_Ps[m] * next.neurons[m][i] * (now.outputs[i] * (1 - now.outputs[i]))
                new_Ps.append(P_m)

                for w in range(len(now.neurons[i])):
                    if layer != 0:
                        now.neurons[i][w] -= (self.learning_rate * P_m * prev.outputs[w])
                    elif layer == 0:
                        now.neurons[i][w] -= (self.learning_rate * P_m * self.input[index][w]) # if first layer
            prev_layer_Ps = copy.deepcopy(new_Ps)

    def compute(self, epoch):# learning
        
        for i in range(epoch):
            index = random.randrange(0, len(self.input))
            self.forwardpass(self.input[index])
            self.backpropagation(index)

    def testing(self, input):
        self.forwardpass(input)
        return self.layers[-1].outputs[0]


if __name__ == "__main__":
    train_file = open('Trn.txt', 'r')
    test_file = open('Tst.txt', 'r')
    train_points = read_file(train_file)
    result_list = []
    for l in train_points:
        result_list.append(l.pop())

    test_points = read_file(test_file)

    mlp = MLP(train_points, result_list, 1, 0.1, 3, [10,10,1], 1000000) # 0.1, 3, [10,10,1] default

    test_result = []
    #for i in test
    for l in test_points:
        temp = mlp.testing(l)
        print(temp)
        if temp > 0.5:
            test_result.append(1)
        else:
            test_result.append(0)
        
    df_trn = pd.DataFrame(train_points, columns = ["x1", "x2"])
    df_y = pd.DataFrame(result_list, columns = ["y"])
    df_tst = pd.DataFrame(test_points, columns = ["x1", "x2"])
    df_tst_y = pd.DataFrame(test_result, columns = ["y"])

    plt.scatter(df_tst['x1'], df_tst['x2'], c=df_tst_y['y'])
    plt.savefig('test.png')
    plt.scatter(df_trn['x1'], df_trn['x2'], c=df_y['y'], cmap='rainbow')
    plt.savefig('train.png')
    