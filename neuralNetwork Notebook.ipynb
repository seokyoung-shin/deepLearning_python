import numpy as np
import scipy.special # 시그모이드 함수를 사용하기 위한 라이브러리
import scipy.misc
import matplotlib.pyplot

%matplotlib inline

   
# 신경망 클래스 정의
class neuralNetwork:
    
    # 신경망 초기하화기
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 입력, 은닉, 출력 계층의 노드 개수 설정
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # 학습률
        self.lr = learningrate
        
        # 가중치 행렬 wij 는 노드 i에서 다음 계층의 노드 j로 연결됨을 의미
        # 입력, 은닉 계층 사이의 가중치 행렬
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        # 은닉, 출력 계층 사이의 가중치 행렬
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        
        # 활성화 함수로 시그모이드 함수를 이용
        # lambda는 익명함수 선언을 의미함
        self.activation_function = lambda x: scipy.special.expit(x)
        
        # 역활성화 함수로 로지트 함수를 이용
        self.inverse_activation_function = lambda x: scipy.special.logit(x)

        pass
    
      
    # 신경망 학습시키기(2단계 존재)
    # 1. 주어진 학습 데이터에 대해 결과 값을 계산하는 단계
    # 2. 계산한 결과값을 실제의 값과 비교하여 이 차이를 이용해 가중치를 업데이트하는 단계 
    def train(self, inputs_list, targets_list):
        # 입력 리스트를 2차원의 행렬로 변환
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        # 은닉 계층으로 들어오는 신호를 계산
        # numpy.dot은 행렬곱을 계산하는 함수
        hidden_inputs = np.dot(self.wih, inputs)
        # 은닉 계층에서 나가는 신호를 계산 - 입력값에 시그모이드 함수를 적용해줌
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # 최종 출력 계층으로 들어오는 신호를 계산
        final_inputs = np.dot(self.who, hidden_outputs)
        # 최종 계층에서 나가는 신호를 계산 - 입력값에 시그모이드 함수를 적용해줌
        final_outputs = self.activation_function(final_inputs)
                        
        
        # 출력 계층의 오차는 (실제값 - 계산값)
        output_errors = targets - final_outputs
        # 은닉 계층의 오차는 가중치에 의해 나뉜 출력 계층의 오차들을 재조잡해 계산
        hidden_errors = np.dot(self.who.T, output_errors)
        
        # 은닉 계층과 출력 계층 간의 가중치 업데이트
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        
        # 입력 계층과 은닉 계층 간의 가중치 업데이트
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        pass
    
    # 신경망 질의하기
    # 신경망으로 들어오는 입력을 받아 출력을 반환해 준다.
    def query(self, input_list):
        # 입력 리스트를 2차원 행렬로 변환
        inputs = np.array(input_list, ndmin=2).T
        
        # 은닉 계층으로 들어오는 신호를 계산
        # numpy.dot은 행렬곱을 계산하는 함수
        hidden_inputs = np.dot(self.wih, inputs)
        
        # 은닉 계층에서 나가는 신호를 계산 - 입력값에 시그모이드 함수를 적용해줌
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # 최종 출력 계층으로 들어오는 신호를 계산
        final_inputs = np.dot(self.who, hidden_outputs)
        # 최종 계층에서 나가는 신호를 계산 - 입력값에 시그모이드 함수를 적용해줌
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
    
    # 신경망 역질의하기(backwards query)
    def backquery(self, label_input):
        
        label = label_input
        # create the output signals for this label
        targets_list = np.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets_list[label] = 0.99
        print(targets_list)
        
        final_outputs = np.array(targets_list, ndmin=2).T
        
        # calculate the signal into the final output layer
        final_inputs = self.inverse_activation_function(final_outputs)

        # calculate the signal out of the hidden layer
        hidden_outputs = np.dot(self.who.T, final_inputs)
        # scale them back to 0.01 to .99
        hidden_outputs -= np.min(hidden_outputs)
        hidden_outputs /= np.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01
        
        # calculate the signal into the hidden layer
        hidden_inputs = self.inverse_activation_function(hidden_outputs)
        
        # calculate the signal out of the input layer
        inputs = np.dot(self.wih.T, hidden_inputs)
        # scale them back to 0.01 to .99
        inputs -= np.min(inputs)
        inputs /= np.max(inputs)
        inputs *= 0.98
        inputs += 0.01
        
        matplotlib.pyplot.imshow(inputs.reshape(28,28), cmap='Greys', interpolation='None')
        pass
    
    
# 훈련 데이터로 훈련시키기 위한 모듈    
def train_neural():
    print("데이터 훈련 시작")
    
    #mnist 학습 데이터인 csv 파일을 리스트로 불러오기 
    training_data_file = open("mnist_dataset/mnist_train.csv", "r")
    training_data_list = training_data_file.readlines()
    #print(training_data_list[0])
    training_data_file.close()

    # 신경망 학습시키기 
    # 학습 데이터 모음 내의 모든 레코드 탐색
    for record in training_data_list:
        all_values = record.split(',')

        # 입력값들이 0.01 ~ 1.0 의 값을 갖도록 조정(0 ~ 255 -> 0.01 ~ 1.0)
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

        # 결과값 생성(실제(정답) 값인 0.99 외에는 모두 0.01로 설정)
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        
        # 신경망 학습
        n.train(inputs, targets)
    
    print("데이터 훈련 끝")
    pass
    
# 테스트 데이터로 신경망 테스트 하기 
def test_neural():
    
    #mnist 테스트 데이터인 csv 파일을 리스트로 불러오기 
    test_data_file = open("mnist_dataset/mnist_test.csv", "r")
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    
    scored = []
    
    # 테스트 데이터 내의 모든 레코드 탐색
    for record in test_data_list:
        all_values = record.split(",")
#         print("정답은 : " , int(all_values[0]))
        
        # 입력 값 범위 조정((0 ~ 255 -> 0.01 ~ 1.0))
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        
        # 신경망에 질의
        outputs = n.query(inputs)
        max_index = np.argmax(outputs)
#         print("신경망의 판단은 : ", max_index)
        
        if max_index == int(all_values[0]):
            scored.append(1)
        else :
            scored.append(0)
            
    scored_array = np.asarray(scored)
    print("정확도 : ", scored_array.sum() / scored_array.size)
    pass

# 이미지 파일을 읽어서, 신경망에 넣을 수 있는 형태로 변경하기
def read_image_file(fileName):
    label = int(fileName[-5:-4])
    img_array = scipy.misc.imread(fileName, flatten=True)
    
    img_data  = 255.0 - img_array.reshape(784)
    # then scale data to range from 0.01 to 1.0
    img_data = (img_data / 255.0 * 0.99) + 0.01
    
    record = np.append(label,img_data)
    matplotlib.pyplot.imshow(record[1:].reshape(28,28), cmap='Greys', interpolation='None')
    print(n.query(record[1:]))
    
    
    
# 입력, 은닉, 출력 노드의 수   
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

# 학습률
learning_rate = 0.2

# 신경망의 인스턴스 생성
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)