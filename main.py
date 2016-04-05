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

    @classmethod
    def make_neuron(cls, make_list):
        for name in make_list:
            cls(name)


# 입력 뉴런
class InputLayer(Neuron):
    all_neuron = list()

    def __repr__(self):
        return "<Input Neuron %s>" % self.name

    @classmethod
    def connect_conjunction(cls):
        for input_neuron in cls.all_neuron:
            for conjunction_neuron in ConjunctionLayer.all_neuron:
                input_neuron.add_next_neuron(conjunction_neuron)


# 결합 뉴런
class ConjunctionLayer(Neuron):
    all_neuron = list()

    def __repr__(self):
        return "<Combine Neuron %s>" % self.name

    @classmethod
    def connect_output(cls):
        for conjunction_neuron, output_neuron in zip(cls.all_neuron, OutputLayer.all_neuron):
            conjunction_neuron.add_next_neuron(output_neuron)


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

    print()
