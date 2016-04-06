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
    @classmethod
    def next_step(cls):
        for neuron in cls.all_neuron:
            for next_neuron, weight in neuron.next_neuron:
                next_neuron.data += neuron.data * weight


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
            self.ConjunctionLayer[conjunctionlayer_name].add_next_neuron(self.OutputLayer[outputlayer_name])

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

        print(1)

    def train(self, train_data):
        """
        Train weight of Neural Connect
        :param train_data: [**input, result]
        """

        """
        Example: train([1,0,1,1,-1,3]
        일 때,
            input1 = 1 ( 날개 있어요 )
            input2 = 0 ( 꼬리 모르겠어요 )
            input3 = 1 ( 부리 있어요 )
            input4 = 1 ( 깃털 있어요 )
            input5 = -1 ( 엔진 없어요 )
            result = 1 ( 두 번째 결과. 여기서는 새)

        """

        self.get_result(train_data)

        return None


if __name__ == '__main__':

    input_list = ['날개', '꼬리', '부리', '깃털', '엔진']
    conjunction_list = ['합규칙1', '합규칙2', '합규칙3']
    output_list = ['비행기', '새', '글라이더']

    model = NeuronModel(input_list, conjunction_list, output_list)

    model.train([1, 0, 1, 0, 1, 2])

    answer_list = {'yes': 1, 'no': -1, 'dunno': 0}

    # 입력 뉴런에 값을 입력함
    for inputlayer_name in model.InputLayer:
        question = "%s가 있나요? yes/no/dunno : " % model.InputLayer[inputlayer_name].name
        model.InputLayer[inputlayer_name].data = answer_list[input(question)]

    print()
