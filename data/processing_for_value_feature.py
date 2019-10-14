import random 
import math
import json

feature_flag = [False] * 13 + [True] * 26


def criteos_way(x):
	# print x
	if float(x) >= 2:
		x = math.log(float(x)) ** 2
	return float(x)


def _value_change(x):
	for idx in range(39):
		if not feature_flag[idx]:
			if x[idx] != u'<empty>':
				x[idx] = criteos_way(x[idx])
			else:
				x[idx] = 0.
	return x

# for idx in range(10):
# 	_filename = "./no_process_real_value/train_" + str(idx) + '.json'
# 	print _filename
# 	with open(_filename, 'r') as f:
# 		data = json.load(f)
# 		# print data[0]
# 		# for idx, _data in enumerate(data):
# 		# 	print _data[0]
# 		label, feature = zip(*data)
# 		feature = map(_value_change, feature)
# 		data = zip(*(label, feature))


# 	with open("./train_" + str(idx) + '.json', 'w') as f:
# 		f.write(json.dumps(data))


# _filename = "./no_process_real_value/valid.json"
# print _filename
# with open(_filename, 'r') as f:
# 	data = json.load(f)
# 	# print data[0]
# 	# for idx, _data in enumerate(data):
# 	# 	print _data[0]
# 	label, feature = zip(*data)
# 	feature = map(_value_change, feature)
# 	data = zip(*(label, feature))


# with open("./valid.json", 'w') as f:
# 	f.write(json.dumps(data))


_filename = "./no_process_real_value/each_feature_id.json"
print _filename
with open(_filename, 'r') as f:
	data = json.load(f)
	for idx in range(len(data)):
		if not feature_flag[idx]:
			data[idx] = {"<empty>": 0} 
	

with open("./each_feature_id.json", 'w') as f:
	f.write(json.dumps(data))