import random


class Neuron:
    all_neuron = list()

    def __init__(self, name=""):
        self.name = name
        self.data = 0
        self.next_neuron = list()
        self.before_neuron = list()

        self.__class__.all_neuron.append(self)

    def __repr__(self):
        return "<Neuron %s>" % self.name

    def add_next_neuron(self, neuron, weight=None):
        if weight is None:
            weight = random.random()

        # Add next neuron at self
        self.next_neuron.append([neuron, weight])

        # Add self to before neuron at next neuron
        neuron.before_neuron.append([self, weight])

    # 다음 단계 뉴런으로 넘어감
    def next_step(self):
        for next_neuron, weight in self.next_neuron:
            next_neuron.data += self.data * weight


# 입력 뉴런
class InputLayer(Neuron):
    all_neuron = list()

    def __repr__(self):
        return "<Input Neuron %s>" % self.name


# 결합 뉴런
class ConjunctionLayer(Neuron):
    all_neuron = list()

    def __repr__(self):
        return "<Combine Neuron %s>" % self.name


# 출력 뉴런
class OutputLayer(Neuron):
    all_neuron = list()

    def __repr__(self):
        return "<Output Neuron %s>" % self.name


class NeuronModel:
    @staticmethod
    def setting_layer_dict(input_name_list, conjunction_name_list, output_name_list):

        outputlayer_dict = dict()
        conjunctionlayer_dict = dict()
        inputlayer_dict = dict()

        # 각 뉴런 갯수가 다를 수 있으므로 별도 반복문을 이용해 처리한다. (input_name_list, conjunction_name_list, output_name_list)
        # 각 layer dict 에 자신의 이름을 key 로 가지는 값을 넣는다.

        for input_name in input_name_list:
            inputlayer_dict[input_name] = InputLayer(input_name)

        for conjunction_name in conjunction_name_list:
            conjunctionlayer_dict[conjunction_name] = ConjunctionLayer(conjunction_name)

        for output_name in output_name_list:
            outputlayer_dict[output_name] = OutputLayer(output_name)

        return inputlayer_dict, conjunctionlayer_dict, outputlayer_dict

    def __init__(self, input_name_list, conjunction_name_list, output_name_list):

        self.InputLayer, self.ConjunctionLayer, self.OutputLayer = \
            self.setting_layer_dict(input_name_list, conjunction_name_list, output_name_list)

        # 입력 뉴런과 결합 뉴런 연결
        self.connect_conjunction()

        # 결합 뉴런과 출력 뉴런 연결
        self.connect_output()

    # 결합 뉴런과 N:N 으로 연결
    def connect_conjunction(self):
        for inputlayer_name in self.InputLayer:
            for conjunctionlayer_name in self.ConjunctionLayer:
                self.InputLayer[inputlayer_name].add_next_neuron(self.ConjunctionLayer[conjunctionlayer_name])

    # output 뉴런과 1:1 로 연결
    def connect_output(self):
        for conjunctionlayer_name, outputlayer_name in zip(self.ConjunctionLayer, self.OutputLayer):
            self.ConjunctionLayer[conjunctionlayer_name].add_next_neuron(self.OutputLayer[outputlayer_name], 1)

    def get_result(self, input_list):
        """
        :param input_list: 입력값과 결과값을 리스트로 받음
        :return: 결과값 (출력뉴런 데이터. 비행기, 새..etc)
        """

        # 모든 입력 뉴런 마다
        for inputlayer_name, input_data in zip(self.InputLayer, input_list):
            neuron = self.InputLayer[inputlayer_name]

            # 초기값 입력
            neuron.data = input_data

            # 다음 단계 뉴런으로 가중치를 곱한 데이터 전달
            neuron.next_step()

        # 모든 결합 뉴런 마다
        for conjunction_name in self.ConjunctionLayer:
            # 출력 뉴런으로 데이터 전달
            self.ConjunctionLayer[conjunction_name].next_step()

        # 모든 결합 뉴런 중, data 가 가장 큰 뉴런 객체를 반환합니다
        return max(OutputLayer.all_neuron, key=lambda output_neuron: output_neuron.data)

    def train(self, train_data):
        """
        Train weight of Neural Connect
        :param train_data: [**input, result]
        """

        """
        Example: train([1,0,1,1,-1,1])
        일 때,
            input1 = 1 ( 날개 있어요 )
            input2 = 0 ( 꼬리 모르겠어요 )
            input3 = 1 ( 부리 있어요 )
            input4 = 1 ( 깃털 있어요 )
            input5 = -1 ( 엔진 없어요 )
            result = 1 ( 두 번째 결과. 여기서는 새)

        result 값
            0: 비행기
            1: 새
            2: 글라이더
            리스트 index 값과 같다.
        """

        # train_data[-1] ---> train_result (위 주석에서 언급한 result 값)
        try:
            train_neuron = OutputLayer.all_neuron[train_data[-1]]
        except IndexError:
            # 학습 데이터에서 빈 줄이 왔을 경우
            # len(train_data) == 0
            return None

        # 실제 가중치 계산을 통해 도출된 출력 뉴런
        result_neuron = self.get_result(train_data)

        if train_neuron != result_neuron:
            self.fix_weight(train_neuron, result_neuron)

    def fix_weight(self, train_result_neuron, result_neuron):
        """
        :param train_result_neuron: 학습 데이터의 출력 뉴런
        :param result_neuron: 실제 계산 후 도출된 출력 뉴런

        학습 데이터 출력 뉴런의 가중치는 높이고,
        (잘못된 결과가 나왔을 때) result_neuron 에 대한 입력 뉴런 가중치는 낮춘다.

        """

        # 학습 데이터 출력 뉴런에 대한 가중치 증가

        # 출력 뉴런에 연결되어 있는 모든 결합 뉴런에 대해
        for conjunction_neuron, conjunction_neuron_weight in train_result_neuron.before_neuron:
            # 결합 뉴런에 연결되어 있는 모든 입력 뉴런에 대해
            input_neuron_list = conjunction_neuron.before_neuron
            for index in range(len(input_neuron_list)):
                input_neuron, weight = input_neuron_list[index]

                # 비활성화 된 입력 뉴런은 넘어감
                if input_neuron.data == 0:
                    continue

                # 활성화 된 입력 뉴런만
                else:
                    # 가중치 40% 증가
                    input_neuron_list[index][1] = (input_neuron_list[index][1] * 1.6)

        # 잘못된 출력 뉴런 결과값에 대한 입력 뉴런 가중치 감소

        # 출력 뉴런에 연결되어 있는 모든 결합 뉴런에 대해
        for conjunction_neuron, conjunction_neuron_weight in result_neuron.before_neuron:
            # 결합 뉴런에 연결되어 있는 모든 입력 뉴런에 대해
            input_neuron_list = conjunction_neuron.before_neuron
            for index in range(len(input_neuron_list)):
                input_neuron, weight = input_neuron_list[index]

                # 비활성화 된 입력 뉴런은 넘어감
                if input_neuron.data == 0:
                    continue

                # 활성화 된 입력 뉴런만
                else:
                    # 가중치 40% 감소
                    input_neuron_list[index][1] = (input_neuron_list[index][1] * 0.4)


if __name__ == '__main__':

    input_list = ['날개', '꼬리', '부리', '깃털', '엔진']
    conjunction_list = ['합규칙1', '합규칙2', '합규칙3']
    output_list = ['비행기', '새', '글라이더']

    model = NeuronModel(input_list, conjunction_list, output_list)

    with open('train_data.txt', 'r') as f:
        all_train_data = [[int(data) for data in train_data.split()] for train_data in f.readlines()]

    for train_data in all_train_data:
        model.train(train_data)

    answer_list = {'yes': 1, 'no': -1, 'dunno': 0}

    input_data = [answer_list[input(model.InputLayer[inputlayer_name].name + "가 있나요? yes/no/dunno : ")]
                  for inputlayer_name in model.InputLayer]

    print("결과 : " + model.get_result(input_data).name)
