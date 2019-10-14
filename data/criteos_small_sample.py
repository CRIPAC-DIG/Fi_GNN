# criteos
import random
import json

with open('old/train_0.json') as f:
    train = json.load(f)

train_length = len(train)

small_train_length = int(train_length / 20.)

print "len_train", small_train_length

data_train = random.sample(train, small_train_length)


data_split_train = data_train[0:int(small_train_length* 0.8)]
data_split_valid = data_train[int(small_train_length* 0.8):]

data_train = data_split_train
valid = data_split_valid

with open('small/train_0.json', 'w') as f:
    f.write(json.dumps(data_train))

with open('old/each_feature_id.json', 'r') as f:
    old_feature_id = json.load(f)

each_feature_id = zip(*data_train)[1]

each_feature_id = zip(*each_feature_id)

for idx in range(len(each_feature_id)):
    temp_list = list(set(each_feature_id[idx]))
    each_feature_id[idx] = dict(zip(*(temp_list, range(len(temp_list)))))
# for idx, att in enumerate(attribute_list):
#     final_feature_id.append(dict(zip(*(each_feature_id[idx], range(len(each_feature_id[idx]))))))
with open('small/each_feature_id.json', 'w') as f:
    f.write(json.dumps(each_feature_id))
    
# with open('old/valid.json') as f:
#     valid = json.load(f)

print "original valid", len(valid)
def _filter_feature(x):
    feature = x[1]
    for idx, each_feature in enumerate(feature):
        if not each_feature_id[idx].has_key(each_feature):
            return False
    return  True

valid_new = filter(_filter_feature, valid)
print "new valid", len(valid_new)

# valid_new = random.sample(valid_new, int(len(valid_new) / 100.))



# print "new valid", len(valid_new)



with open("small/valid.json", 'w') as f:
    f.write(json.dumps(valid_new))




