import random
from konlpy.tag import Mecab


class Neuron:
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


def language_processing(input_data):
    mecab = Mecab()

    # 명사에 대한 yn 데이터 저장
    # 날개가 있을 경우, check_data['날개'] == 1
    check_data = dict()
    for name in [input_neuron.name for input_neuron in InputLayer.all_neuron]:
        # 우선 check_data 의 모든 데이터를 모른다는 조건으로 초기화
        check_data[name] = 0

    # [*range(3)] is same with [0, 1, 2]
    word_list, pos_list = zip(*[(word, pos)
                                for word, pos in mecab.pos(input_data)
                                if pos in ['VV', 'VA', 'NNG', 'JC', 'SC', 'MAG', 'VX']])

    # 이미 처리한 word 데이터를 False 로 바꾸기 위해
    # 데이터 변경을 지원하는 리스트로 형 변환. (기존에는 tuple)
    word_list = list(word_list)

    # 같은 이유
    pos_list = list(pos_list)

    # 부정적인 성분 부사를 가지고 있는 형용사를 치환
    # 날개가 안 보인다 --> 날개가 없다

    yn_dict = {
        '있': 1,
        '들리': 1,
        '보이': 1,
        '없': -1,
        '모르': 0
    }

    """
    for index in range(len(pos_list)):
        if pos_list[index] == 'MAG' and word_list[index] == '안':  # 성분 부사 이면서 부정 부사 일 경우
            word_list[index] = '없'  # 부정으로 치환


        for i in range(len(pos_list[index:])):  # 부정 부사 뒷 부분 탐색
            if pos_list[i] in ['VV', 'VA']:  # '있', '없' 등의 데이터가 나올 경우
                try:
                    word_list[i] = yn_change[word_list[i]]  # yn_change 를 이용해 반전시킨다
                except KeyError:
                    word_list
                    pass
    """

    # 형용사를 먼저 탐색하고, 주변 명사를 그룹화 하는 방식으로 처리한다.

    # pos 데이터 중에서 있,없 등의 수식어를 가져옴
    for index in range(len(pos_list)):
        if pos_list[index] == 'MAG' and word_list[index] == '안':  # 성분 부사 이면서 부정 부사 일 경우
            word_list[index] = '없'  # 부정으로 치환
            pos_list[index] = 'VA'  # pos 데이터도 맞게 변경

        if pos_list[index] in ['VA', 'VV']:  # if pos is yn data

            # 해당 명사에 서술한 내용에 따라 InputLayer Neuron 에 입력함
            try:
                yn = yn_dict[word_list[index]]
            except KeyError:
                yn = 0
            finally:
                # 뒤에 부정적인 보조용언이 올 경우
                # ex) ~하지 '않'는다

                # 다음 인덱스 부터 탐색
                tmp_index = index + 1
                while tmp_index < len(pos_list):
                    if pos_list[tmp_index] == 'VX':
                        if word_list[tmp_index] == '않':
                            yn *= -1
                            break
                    tmp_index += 1

            # 그 전까지의 모든 명사를 위 yn 데이터로 저장
            for nng in [word_list[i] for i in range(index) if pos_list[i] == 'NNG']:
                # 이미 처리한 word 일 경우
                if nng is False:
                    continue
                else:
                    try:
                        check_data[nng]
                    except KeyError:
                        pass
                    else:
                        check_data[nng] = yn

            # 처리한 word 들은 False 으로 치환.
            word_list[:index] = ([False] * index)

    return check_data


if __name__ == '__main__':

    input_list = ['날개', '꼬리', '부리', '깃털', '엔진']  # VA, NNG
    conjunction_list = ['합규칙1', '합규칙2', '합규칙3']
    output_list = ['비행기', '새', '글라이더']

    model = NeuronModel(input_list, conjunction_list, output_list)

    with open('train_data.txt', 'r') as f:
        all_train_data = [[int(data) for data in train_data.split()] for train_data in f.readlines()]

    for train_data in all_train_data:
        model.train(train_data)

    # 학습 끝

    data = "엔진소리가 들리지 않고, 꼬리가 보이지 않는다. 부리가 없다. 또한 깃털도 안보인다"
    data2 = "엔진과 날개가 있으며  꼬리는 모르겠다. 부리가 안보인다. 아니다, 꼬리가 있다."
    data3 = "글라이더 같은 데, 잘 알 수 없고 꼬리가 없다. 엔진 소리가 안 들리고 날개가 있다"

    for i in [data, data2, data3]:
        print("")
        print(i)
        print(language_processing(i))
        print("")

    nl_data = language_processing(input("특징을 입력해주세요 : "))
    print(nl_data)
    input_data = [nl_data[name] for name in input_list]

    result = model.get_result(input_data)

    print(result.name)
