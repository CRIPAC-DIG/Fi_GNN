import json
from collections import Counter

data_dict_train = {}
attr_list = []

with open('train.csv', 'r') as f:
    for idx, line in enumerate(f):
        if idx % 100000 == 0:
            print idx
        if idx == 0:
            attr_list = line.strip().split(',')
            for jdx, attribute in enumerate(attr_list):
                data_dict_train[attribute] = []
        else:
            temp = line.strip().split(',')
            for jdx, feature in enumerate(temp):
                data_dict_train[attr_list[jdx]].append(temp[jdx])

with open('alltraindata.json', 'w') as f:
    f.write(json.dumps(data_dict_train))
    # data_dict_train = json.load(f)
attr_list = data_dict_train.keys()
# print attr_list

print len(data_dict_train['device_id'])
# user_id = zip(*(data_dict_train['device_id'], data_dict_train['device_ip']))
user_id = data_dict_train['device_id']
count_user = Counter(user_id)
print "user_num", len(count_user)
# arrange by user
data_listby_user = []
user2id = dict(zip(*(count_user.keys(), range(len(count_user)))))
# user_idx = 2
user_list = count_user.keys()
with open("user_list.json", 'w') as f:
    f.write(json.dumps(user_list))


print "all_user_len", len(user_list)
little_user_list = user_list[user_idx * int(len(user_list)/10):(user_idx + 1) * int(len(user_list)/10)]
user2id = dict(zip(*(little_user_list, range(len(little_user_list)))))
print "little_user_len", len(little_user_list)

data_listby_user = []
for _ in user2id:
    data_listby_user.append([])
for idx in range(len(data_dict_train['device_id'])):
    # data_listby_user[user2id[(data_dict_train['device_id'][idx], data_dict_train['device_ip'][idx])]].append(
    # tuple([data_dict_train[attr][idx] for attr in attr_list]))
    if user2id.has_key(data_dict_train['device_id'][idx]):
        # print idx
        # print user2id[data_dict_train['device_id'][idx]]
        # print tuple([data_dict_train[attr][idx] for attr in attr_list])
        data_listby_user[user2id[data_dict_train['device_id'][idx]]].append(
        tuple([data_dict_train[attr][idx] for attr in attr_list]))
        if idx % 100000 == 0:
            print idx



data_listby_user = filter(lambda x: len(x) > 3, data_listby_user)
#
# all_length = len(data_listby_user)
# print "new_user_num", all_length

# with open('all_data_sequence.json', 'w') as f:
#     f.write(json.dumps(data_listby_user))
# for idx in xrange(10):
print user_idx
all_length = len(data_listby_user)
for user_jdx in range(10):
    with open('data_listby_user_' + str(user_idx) + '_' + str(user_jdx) + '.json', 'w') as f:
        # f.write(json.dumps(data_listby_user[int(idx * (all_length * 0.1)): int((idx + 1) * (all_length * 0.1))]))
        f.write(json.dumps(data_listby_user[int(user_jdx * (all_length * 0.1)): int((user_jdx + 1) * (all_length * 0.1))]))
