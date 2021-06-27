import pickle


def merge_maps(dict1, dict2):
    """用于合并两个word2id或者两个tag2id"""
    for key in dict2.keys():
        if key not in dict1:
            dict1[key] = len(dict1)
    return dict1


def save_model(model, file_name):
    """用于保存模型"""
    with open(file_name, "wb") as f:
        pickle.dump(model, f)


def load_model(file_name):
    """用于加载模型"""
    with open(file_name, "rb") as f:
        model = pickle.load(f)
    return model


# 读取txt里的内容到list
def read_txt_to_list():
    res = []
    file = open('resume.txt', 'r')
    while 1:
        char = file.read(1)
        if not char:
            break
        res.append(char)
    file.close()
    return res

# 整合全部结果
def format_res(word_res, tag_res):
    w = []
    t = []
    for i in range(len(word_res)):
        w_res, t_res = con_res(word_res[i], tag_res[i])
        w.append(w_res)
        t.append(t_res)
    return w, t


# 整合结果，只针对某一个人的结果
def con_res(pred_word_list, pred_tag_list):
    formatted_tag_list = []
    formatted_word_list = []
    for i in range(len(pred_tag_list)):
        if pred_tag_list[i].startswith('O'):  # 如果这个标签是无意义词
            formatted_tag_list.append("NULL")  # 则标签结果加上NULL
            formatted_word_list.append(pred_word_list[i])  # word结果加上这个字符本身
            continue
        if pred_tag_list[i].startswith('B'):  # 如果这是一个标签的开始
            str_tag = pred_tag_list[i][2:]  # 标签
            formatted_tag_list.append(str_tag)

            str_word = pred_word_list[i]  # 对应位置的字符就是字符的开始
            j = i + 1
            while 1:
                if pred_tag_list[j].startswith('E'):
                    str_word += pred_word_list[j]
                    break
                else:
                    str_word += pred_word_list[j]
                    j += 1
            formatted_word_list.append(str_word)
    return formatted_word_list, formatted_tag_list


# 按照B-M-E的格式修改tag list的格式
def format_tag_list(tag_list):
    # 把tag list标签的最后一个改成E开头，中间的改成M开头
    if tag_list[-1].startswith('I'):
        tag_list[-1] = 'E' + tag_list[-1][1:]

    for i in range(len(tag_list) - 1):
        if tag_list[i].startswith('I'):
            if tag_list[i + 1].startswith('I') or tag_list[i + 1].startswith('E'):  # 后面那个以I或E开头，说明还在内部
                tag_list[i] = 'M' + tag_list[i][1:]
            else:  # 其他情况，说明这个标签结束了
                tag_list[i] = 'E' + tag_list[i][1:]


# 修改简历格式，现有的格式以"_换行符_"作为两个人的分界
# 输出两个list（word和tag），分别由子list组成，子list即为一个人的信息，如：
# word_list = [[name1, org1], [name2, org2], ...]
# 注意：最后一个人结束后也需要输入"_换行符_"，否则他的信息将丢失！！！
def generate_resume_list_from_txt(filename):
    whole_word_list = []
    whole_tag_list = []
    one_person_word_list = []
    one_person_tag_list = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line[0:2] == '_换':  # 这一行是换行符，一个人结束，将他的list加入总list，并刷新子list
                whole_word_list.append(one_person_word_list)
                whole_tag_list.append(one_person_tag_list)
                one_person_word_list = []
                one_person_tag_list = []
            else:
                word = line[0]  # 内容（一个字）
                tag = line[2]  # 标签的开端
                i = 3
                while 1:
                    if line[i] == '\t':  # 从标签的开端开始，直到'\t'，说明标签结束
                        break
                    else:
                        tag += line[i]
                    i += 1
                if len(tag) > 1:  # 长度>1，说明是实体标签，需要修改格式，否则是'0'，不需要修改
                    formatted_tag = tag[0] + '-' + tag[5:].upper()
                else:
                    formatted_tag = tag
                one_person_word_list.append(word)
                one_person_tag_list.append(formatted_tag)

    # 逐个修改tag的格式
    for person in whole_tag_list:
        format_tag_list(person)
    return whole_word_list, whole_tag_list


# 生成tag set
def generate_tag_set():
    _, tag_list = generate_resume_list_from_txt("data_for_train.txt")
    flatten_tag_list = flatten_lists(tag_list)
    tag_set = set()
    for tag in flatten_tag_list:
        tag_set.add(tag)
    return tag_set


# LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
# 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
def extend_maps(word2id, tag2id, for_crf=True):
    word2id['<unk>'] = len(word2id)
    word2id['<pad>'] = len(word2id)
    tag2id['<unk>'] = len(tag2id)
    tag2id['<pad>'] = len(tag2id)
    # 如果是加了CRF的bilstm  那么还要加入<start> 和 <end>token
    if for_crf:
        word2id['<start>'] = len(word2id)
        word2id['<end>'] = len(word2id)
        tag2id['<start>'] = len(tag2id)
        tag2id['<end>'] = len(tag2id)

    return word2id, tag2id


def prepocess_data_for_lstmcrf(word_lists, tag_lists, test=False):
    assert len(word_lists) == len(tag_lists)
    for i in range(len(word_lists)):
        word_lists[i].append("<end>")
        if not test:  # 如果是测试数据，就不需要加end token了
            tag_lists[i].append("<end>")

    return word_lists, tag_lists


def flatten_lists(lists):
    flatten_list = []
    for l in lists:
        if type(l) == list:
            flatten_list += l
        else:
            flatten_list.append(l)
    return flatten_list
