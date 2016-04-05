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
    def __init__(self, input_name_list, conjunction_name_list, output_name_list):
        self.InputLayer = [InputLayer(input_name) for input_name in input_name_list]
        self.ConjunctionLayer = [ConjunctionLayer(conjunction_name) for conjunction_name in conjunction_name_list]
        self.OutputLayer = [OutputLayer(output_name) for output_name in output_name_list]

        self.connect_conjunction()
        self.connect_output()

    # 결합 뉴런과 N:N 으로 연결
    def connect_conjunction(self):
        for input_neuron in self.InputLayer:
            for conjunction_neuron in self.ConjunctionLayer:
                input_neuron.add_next_neuron(conjunction_neuron)

    # output 뉴런과 1:1 로 연결
    def connect_output(self):
        for conjunction_neuron, output_neuron in zip(self.ConjunctionLayer, self.OutputLayer):
            conjunction_neuron.add_next_neuron(output_neuron)


if __name__ == '__main__':
    input_list = ['날개', '꼬리', '부리', '깃털', '엔진']
    conjunction_list = ['합규칙1', '합규칙2', '합규칙3']
    output_list = ['비행기', '새', '글라이더']

    model = NeuronModel(input_list, conjunction_list, output_list)

    answer_list = {'yes': 1, 'no': -1, 'dunno': 0}

    # 입력 뉴런에 값을 입력함
    for input_neuron in InputLayer.all_neuron:
        question = "%s가 있나요? yes/no/dunno : " % input_neuron.name
        input_neuron.data = answer_list[input(question)]


    print()
