import random


WINDOW_SIZE = 5
PUNCTUATION_MARK = ['x']  # 标点
PUNCTUATION = ['。', '！', '？', '，', '～']
NOUN_MARK = ['n', 'ng', 'nr', 'nrfg', 'nrt', 'ns', 'nt', 'nz']  # 名词
VERB_MARK = ['v', 'vd', 'vg', 'vi', 'vn', 'vq']  # 动词
ADJECTIVE_MARK = ['a', 'ad', 'an', 'ag']  # 形容词
ADVERB_MARK = ['d', 'df', 'dg']  # 副词
ENG_MARK = ['eng']

EMOJI = []

YANWENZI = []

ILLEGAL_WORD = ['考拉', '网易']  # '不过', '因为', '而且', '但是', '但', '所以', '因此', '如果',


RESERVED_MARK = NOUN_MARK + VERB_MARK + ADJECTIVE_MARK + ADVERB_MARK + ENG_MARK # 用于发现新词
ASPECT_MARK = NOUN_MARK + VERB_MARK


def text2seg_pos(seg_pos_text, pattern='[。！？]'):
    """
    经过分词的文档，原始一条用户评论通过指定的标点符号分成多个句子
    """
    seg_list = []  # 保存全部按标点切分的seg
    pos_list = []  # 保存全部按标点切分的pos
    seg_review_list = []  # 用户完整的一条评论
    for line in seg_pos_text:
        line = line.strip()
        line = line.split(' ')
        seg_sub_list = []
        pos_sub_list = []
        cur_review = []
        for term in line:
            word, flag = term.split('/')
            cur_review.append(word)
            if word in pattern:
                seg_sub_list.append(word)
                pos_sub_list.append(flag)
                seg_list.append(list(seg_sub_list))
                pos_list.append(list(pos_sub_list))
                seg_sub_list = []
                pos_sub_list = []
            else:
                seg_sub_list.append(word)
                pos_sub_list.append(flag)
        seg_review_list.append(list(cur_review))

    # temp_dict = {}
    # seg_unique_list = []
    # for i in seg_list:
    #     _s = ''.join(i)
    #     _s = _s[:-1]  # 去掉标点
    #     if _s in temp_dict:
    #         continue
    #     else:
    #         temp_dict[_s] = 0
    #         seg_unique_list.append(i)

    return seg_list, pos_list, seg_review_list


def get_candidate_aspect(seg_list, pos_list, adj_word, stop_word, word_idf):
    """
    输入的数据为用逗号隔开的短句，
    利用开窗口的方式，根据情感词典抽名词得到候选的aspect
    """
    print("利用情感词典抽取候选aspect...")
    aspect_dict = {}
    for i, sentence in enumerate(seg_list):
        for j, word in enumerate(sentence):
            if word in adj_word and pos_list[i][j] in ADJECTIVE_MARK:  # 当前的词属于情感词且词性为形容词
                startpoint = j - WINDOW_SIZE
                startpoint = startpoint if startpoint >= 0 else 0
                for k in range(startpoint, j):
                    if pos_list[i][k] in ASPECT_MARK:
                        if seg_list[i][k] in aspect_dict:
                            aspect_dict[seg_list[i][k]] += 1
                        else:
                            aspect_dict[seg_list[i][k]] = 1

    temp = aspect_dict.items()
    temp = list(filter(lambda x: len(x[0]) > 1, temp))  # 经过词组发现之后，删去一个字的词
    temp = [item[0] for item in temp if item[0] not in stop_word]  # 删去停用词
    temp = [item for item in temp if word_idf[item] != 0]  # 删去IDF值为0的词
    aspect_list = temp
    print("---aspect抽取完成，共抽取到%s个候选aspect---" % (len(aspect_list)))
    return aspect_list


class NSDict:
    """
    用来构建候选集（aspect，opinion，pattern）
    """
    def __init__(self, seg_list, pos_list, raw_aspect_list):
        self.seg_list = seg_list
        self.pos_list = pos_list
        self.raw_aspect_list = raw_aspect_list
        self.ns_dict = {}
        self.aspect_do_not_use = []
        self.opinion_do_not_use = ["最", "不", "很"]
        self.pattern_do_not_use = ["的-", "和-", "和+", "而+", "而-", "又+", "又-", "而且+", "而且-"]

    def _seg2nsd(self, aspect_for_filter):
        for x, clue in enumerate(self.seg_list):
            N_list = []
            S_list = []
            word_list = clue
            for y, word in enumerate(clue):
                if word in aspect_for_filter:
                    N_list.append(y)
                elif self.pos_list[x][y] in ADJECTIVE_MARK:
                    S_list.append(y)
            if N_list and S_list:
                self._make_nsdict(word_list, N_list, S_list)

    def _make_nsdict(self, word_list, N_list, S_list):
        for n in N_list:
            for s in S_list:
                if (1 < n - s < WINDOW_SIZE + 1) or (1 < s - n < WINDOW_SIZE + 1):  # 窗口大小是5
                    if word_list[n] not in self.ns_dict:
                        self.ns_dict[word_list[n]] = {}
                    if word_list[s] not in self.ns_dict[word_list[n]]:
                        self.ns_dict[word_list[n]][word_list[s]] = {}
                    if n > s:
                        patt = ' '.join(word_list[s + 1: n]) + '+'
                    else:
                        patt = ' '.join(word_list[n + 1: s]) + '-'
                    if patt not in self.ns_dict[word_list[n]][word_list[s]]:
                        self.ns_dict[word_list[n]][word_list[s]][patt] = 0.
                    self.ns_dict[word_list[n]][word_list[s]][patt] += 1.

    def _noise_del(self):
        for aspect in self.aspect_do_not_use:
            self._noise(aspect, self.ns_dict)
        for n in self.ns_dict:
            for opinion in self.opinion_do_not_use:
                self._noise(opinion, self.ns_dict[n])
            for s in self.ns_dict[n]:
                for pattern in self.pattern_do_not_use:
                    self._noise(pattern,self.ns_dict[n][s])

    def _noise(self, str, dict):
        if str in dict:
            del dict[str]

    def build_nsdict(self):
        print("stage 1：抽取pair和pattern...")
        self._seg2nsd(self.raw_aspect_list)
        self._noise_del()
        print("\tDone")
        return self.ns_dict


class PairPattSort:
    '''
    Pair-Patt-Count structure
    '''
    def __init__(self, ns_dict):
        self._get_map(ns_dict)

    def _get_map(self, ns_dict):
        '''
        get map: [pair-patt], [patt-pair], [pair](score), [patt](score)

        :param ns_dict: Entity.str { Emotion.str { Pattern.str { Count.int (It's a three-level hash structure)
        :return:
        '''
        pair_list = []
        patt_dict = {}
        patt_pair_map = {}
        pair_patt_map = {}

        aspects = list(ns_dict.keys())
        aspects.sort()

        for n in aspects:
            for s in ns_dict[n]:
                n_s = "{}\t{}".format(n, s)   #这里存的pair是字符串，中间用\t隔开
                pair_list.append(n_s)
                pair_patt_map[n_s] = {}
                for patt in ns_dict[n][s]:
                    if patt not in patt_dict:
                        patt_dict[patt] = 1.0
                    pair_patt_map[n_s][patt] = ns_dict[n][s][patt]
                    if patt in patt_pair_map:
                        patt_pair_map[patt][n_s] = ns_dict[n][s][patt]
                    else:
                        patt_pair_map[patt] = {}
                        patt_pair_map[patt][n_s] = ns_dict[n][s][patt]
        self.patt_pair_map = patt_pair_map
        self.pair_patt_map = pair_patt_map
        self.pair_len = len(pair_list)
        self.patt_len = len(patt_dict)
        self.pair_score = dict([(word, 1.) for i, word in enumerate(pair_list)])
        self.patt_score = patt_dict

    """"正则化，和为score_len"""
    def _norm(self, score_dict, score_len):
        sum_score = 0.
        for s in score_dict:
            sum_score += score_dict[s]
        for s in score_dict:
            score_dict[s] = score_dict[s] / sum_score * score_len
        return score_dict

    def _patt_pair(self):
        for pair in self.pair_patt_map:  # <- 循环遍历每个pair
            value = 0.
            for patt in self.pair_patt_map[pair]:  # <- 每个pair中的pattern出现的个数 * 这个pattern的score，然后求和得到这个pair的分数
                value += self.pair_patt_map[pair][patt] * self.patt_score[patt]
            self.pair_score[pair] = value

    def _pair_patt(self):
        for patt in self.patt_pair_map:  # <- 遍历每个pattern
            value = 0.
            for pair in self.patt_pair_map[patt]:  # <- 每个被pattern修饰的pair出现的个数 * 这个pair的score，然后求和得到这个pattern1的
                value += self.patt_pair_map[patt][pair] * self.pair_score[pair]
            self.patt_score[patt] = value
    def _patt_correct(self):
        self.patt_score['的-'] = 0.0

    def _iterative(self):
        '''
        A complete iteration
        [pair] = [patt-pair] * [patt]
        [patt] = [pair-patt] * [pair]
        :return:
        '''
        self._patt_pair()
        self.pair_score = self._norm(self.pair_score, self.pair_len)
        self._pair_patt()
        self.patt_score = self._norm(self.patt_score, self.patt_len)

    def sort_pair(self):
        print("stage 2：组合排序...")
        for i in range(100):
            self._iterative()
        pair_score = sorted(self.pair_score.items(), key=lambda d: d[1], reverse=True)
        print('\tDone')
        print("---pair抽取完成---")
        return pair_score


def get_aspect_express(seg_review_list, pair_useful):
    """
    抽取原始评论中的aspect作为输入，完整的评论作为输出
    """

    def check_sentence(sentence):
        """
        判断短句是否合法
        """
        _s = ''.join(sentence)
        legal = True
        if len(_s) > 30:
            legal = False
        return legal

    raw_aspect_express = {k: [] for k in pair_useful}  # 用户关于某个观点的一段原始表达
    raw_aspect_express_count = {k: 0 for k in pair_useful}  # 记录某个观点表达出现的次数
    for review in seg_review_list:  # 每个sentence就是一句完整的review

        source = []  # 训练的src
        if review[-1] not in PUNCTUATION:
            review.append('。')
        target = review  # 训练的tgt

        # 对于单个review进行切分
        cur_review = []
        pre_end = 0
        for i, _ in enumerate(review):
            if review[i] in ['。', '！', '？', '，', '～']:
                cur_review.append(review[pre_end:i + 1])
                pre_end = i + 1
            elif i == len(review) - 1:
                cur_review.append(review[pre_end:])

        for sentence in cur_review:  # sentence 是两个标点之间的短句
            if sentence[-1] not in PUNCTUATION:
                sentence.append('。')
            find_opinion_flag = False
            for idx, word in enumerate(sentence):
                if find_opinion_flag:  # 如果在当前的短句中已经找到了一组观点表达就结束对这个短句的搜索
                    break
                if word in pair_useful:  # 当前的word属于aspect
                    # 向前开窗口
                    startpoint = idx - WINDOW_SIZE if idx - WINDOW_SIZE > 0 else 0
                    for i in range(startpoint, idx):  # 寻找opinion word
                        cur_word = sentence[i]
                        if cur_word in pair_useful[word] and sentence[i + 1] == "的":  # eg. 超赞的一款面膜
                            if check_sentence(sentence):
                                raw_aspect_express[word].append(sentence)
                                raw_aspect_express_count[word] += 1
                                find_opinion_flag = True  # 只要找到一个opinion word就算命中一个短句了

                    # 向后开窗口
                    endpoint = idx + WINDOW_SIZE if idx + WINDOW_SIZE < len(sentence) else len(sentence)
                    for i in range(idx + 1, endpoint):
                        cur_word = sentence[i]
                        if cur_word in pair_useful[word]:
                            if check_sentence(sentence):
                                raw_aspect_express[word].append(sentence)
                                raw_aspect_express_count[word] += 1
                                find_opinion_flag = True  # 只要找到一个opinion word就算命中一个短句了

    # 筛选得到保留的aspect
    aspect_express = {}
    for aspect in raw_aspect_express:
        if raw_aspect_express_count[aspect] < 5:
            continue
        aspect_express[aspect] = raw_aspect_express[aspect]

    return aspect_express


def merge_aspect_express(aspect_express, pair_useful):
    """
    对相似的观点表达进行合并, 同时输出最终的aspect_opinion_pair
    """
    aspects = list(aspect_express.keys())
    length = len(aspects)
    aspects.sort()  # 排成字典序
    merged_aspects = [[aspects[0]]]
    merged_express = {}
    opinion_set = []

    def check_is_same(word1, word2):
        """
        判断两个词当中是否存在相同的字
        """
        for i in word1:
            if i in word2:
                return True
        return False

    for i in range(1, length):
        if check_is_same(merged_aspects[-1][-1], aspects[i]):
            merged_aspects[-1].append(aspects[i])
        else:
            merged_aspects.append([aspects[i]])

    for a_list in merged_aspects:

        # 收集全部的形容词
        for i in a_list:
            opinion_set += pair_useful[i]

        _l = ','.join(a_list)
        merged_express[_l] = []
        for i in a_list:
            merged_express[_l] += aspect_express[i]

    opinion_set = set(opinion_set)

    return merged_express, opinion_set


def build_dataset_express(seg_review_list, pair_useful):
    """
    抽取原始评论中的aspect作为输入，完整的评论作为输出
    """
    train_data = []  # 记录训练数据
    for review in seg_review_list:  # 每个sentence就是一句完整的review

        source = []  # 训练的src
        if review[-1] not in PUNCTUATION:
            review.append('。')
        target = review  # 训练的tgt

        # 对于单个review进行切分
        cur_review = []
        pre_end = 0
        for i, _ in enumerate(review):
            if review[i] in ['。', '！', '？', '，', '～']:
                cur_review.append(review[pre_end:i + 1])
                pre_end = i + 1
            elif i == len(review) - 1:
                cur_review.append(review[pre_end:])

        for sentence in cur_review:  # sentence 是两个标点之间的短
            if sentence[-1] not in PUNCTUATION:
                sentence.append('。')
            find_opinion_flag = False
            for idx, word in enumerate(sentence):
                if find_opinion_flag:  # 如果在当前的短句中已经找到了一组观点表达就结束对这个短句的搜索
                    break
                if word in pair_useful:  # 当前的word属于aspect
                    source.append(word)
                    find_opinion_flag = True  # 只要找到一个opinion word就算命中一个短句了

        train_data.append((list(source), target))

    max_source_length = 0
    # 筛选训练数据
    def check_review(item):
        """
        判断当前review是否合法
        """
        source = item[0]
        tgt = item[1]
        legal = True
        _s = ''.join(tgt)
        if len(source) == 0 or len(source) > 5:  # 不含有观点表达或者观点词太多
            legal = False
        unique_source = set(source)
        if len(unique_source) != len(source):
            legal = False
        if len(_s) > 60:
            legal = False
        return legal

    legal_train_data= []
    for item in train_data:
        if check_review(item):
            max_source_length = max(max_source_length, len(item[0]))
            legal_train_data.append(item)

    print('max source length:%s' % max_source_length)
    return legal_train_data


def generate_reviews(aspect_express, num=1000):
    """
    根据候选集合生成假评论
    """
    all_aspect = list(aspect_express.keys())
    print('Aspect:{}'.format(all_aspect))
    print()

    # 根据不同aspect出现的概率分配不同权重
    aspect_length_dict = {}
    for a in aspect_express:
        aspect_length_dict[a] = len(aspect_express[a])
    weight_aspect_list = []
    for aspect in aspect_length_dict:
        weight_aspect_list += [aspect] * aspect_length_dict[aspect]

    res = []
    for _ in range(num):
        num_aspect = random.choice([1, 2, 3, 4, 5, 6])
        review = []
        used_aspect = []
        for _ in range(num_aspect):
            a = random.choice(weight_aspect_list)
            while a in used_aspect:
                a = random.choice(weight_aspect_list)
            used_aspect.append(a)
            a_s = random.choice(aspect_express[a])
            a_s = a_s[:-1] + ['#']  # 丢掉标点，换位#作为切分点
            review += a_s
        res.append(review)

    return res


def fake_review_filter(reviews, opinion_set):
    """
    筛去评论中不像人写的句子：如果同一个形容词重复出现两次就判定为假评论，同时筛去长度超过60的评论
    """
    results = []
    for review in reviews:
        opinion_used = {k: 0 for k in opinion_set}
        flag = True
        for word in review:
            if word in ILLEGAL_WORD:
                flag = False
            if word in opinion_used:
                opinion_used[word] += 1
                if opinion_used[word] >= 2:
                    flag = False
                    # print('Fake:{}'.format(''.join(review)))
                    break
        if flag:
            _s = ''.join(review)
            _s = _s.split('#')  # 最后一个是空字符
            review = ''
            pu = ['，'] * 100 + ['～'] * 20 + ['！'] * 20 + EMOJI + YANWENZI
            random.shuffle(pu)
            for a_s in _s:
                if a_s:
                    review += a_s + random.choice(pu)
            if not review:
                print('error:')
                print(review)
            review = review[:-1] + '。'
            results.append(review)
            print('\t' + review)

    return results






