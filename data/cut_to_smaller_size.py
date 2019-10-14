import json


for idx in range(10):
	if idx == 0:
		continue
	load_file_name = './train_' + str(idx) + '.json'
	print load_file_name
	with open(load_file_name, 'r') as f:
		dataset = json.load(f)
	len_dataset = len(dataset)

	split_len = int(len_dataset / 10)
	for jdx in range(10):
		_file_name = './cut_to_smaller_size/train_' + str(idx) + str(jdx) + '.json'
		with open(_file_name, 'w') as f:
			f.write(json.dumps(dataset[jdx * split_len: (jdx + 1) * split_len]))

