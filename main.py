import random


class NeuronModel:
    def __init__(self):
        self.InputLayer = InputLayer.all_neuron
        self.ConjunctionLayer = ConjunctionLayer.all_neuron
        self.OutputLayer = OutputLayer.all_neuron

    # 결합 뉴런과 N:N 으로 연결
    @staticmethod
    def connect_conjunction():
        for input_neuron in InputLayer.all_neuron:
            for conjunction_neuron in ConjunctionLayer.all_neuron:
                input_neuron.add_next_neuron(conjunction_neuron)

    # output 뉴런과 1:1 로 연결
    @staticmethod
    def connect_output():
        for conjunction_neuron, output_neuron in zip(ConjunctionLayer.all_neuron, OutputLayer.all_neuron):
            conjunction_neuron.add_next_neuron(output_neuron)


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

    # 이름 목록을 인자로 받아 객체를 생성합니다.
    @classmethod
    def make_neuron(cls, make_list):
        for name in make_list:
            cls(name)

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


if __name__ == '__main__':
    InputLayer.make_neuron(['날개', '꼬리', '부리', '깃털', '엔진'])
    ConjunctionLayer.make_neuron(['합규칙1', '합규칙2', '합규칙3'])
    OutputLayer.make_neuron(['비행기', '새', '글라이더'])

    InputLayer.connect_conjunction()
    ConjunctionLayer.connect_output()

    answer_list = {'yes': 1, 'no': -1, 'dunno': 0}

    # 입력 뉴런에 값을 입력함
    for input_neuron in InputLayer.all_neuron:
        question = "%s가 있나요? yes/no/dunno : " % input_neuron.name
        input_neuron.data = answer_list[input(question)]

    InputLayer.next_step()

    print()
