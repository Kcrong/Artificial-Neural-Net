import random


class Neuron:
    all_neuron = list()

    def __init__(self, name=""):
        self.name = name
        self.next_neuron = list()
        self.before_neuron = list()

        self.__class__.all_neuron.append(self)

    def __repr__(self):
        return "<Neuron %s>" % self.name

    def add_next_neuron(self, neuron, weight=random.uniform(0, 1)):

        # Add next neuron at self
        self.next_neuron.append(dict({neuron, weight}))

        # Add self to before neuron at next neuron
        neuron.before_neuron.append(dict({self, weight}))


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
    def __repr__(self):
        return "<Output Neuron %s>" % self.name


if __name__ == '__main__':

    # 뉴런 선언부
    for input_name in ['날개', '꼬리', '부리', '깃털', '엔진']:
        InputLayer(input_name)

    for conjunction_name in ['새', '비행기', '글라이더']:
        ConjunctionLayer(conjunction_name)

    # 각 입력 뉴런 마다
    for neuron in InputLayer.all_neuron:

        # 각 결합 뉴런 마다
        for conjunction_neuron in ConjunctionLayer.all_neuron:
            # 서로 연결
            neuron.add_next_neuron(conjunction_neuron)


    print()
