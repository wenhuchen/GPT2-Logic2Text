import torch
import json
import urllib.parse

for split in ['train', 'valid', 'test']:
	with open(f'dataset/{split}.json', 'r') as f:
		data = json.load(f)

	examples = {}
	for f in data:
		sentence = f['sent']
		sentence = f['sent'].replace('\u2013', '-')
		annotations = f['annotation']
		table_name = f['url'].split('/')[-1]
		title  = f['wiki'].split('/')[-1]
		title = urllib.parse.unquote(title).replace('_', ' ').lower()
		title = title.replace('\u2013', '-')
		columns = []
		for k, v in annotations.items():
			if 'col' in k:
				if v != 'n/a':
					v = v.split(',')
					columns = columns + [int(_) - 1 for _ in v if int(_) >= 1]
		columns = sorted(columns)

		examples[table_name] = examples.get(table_name, []) + [[sentence, columns, title, 'None']]

	with open(f'data/{split}_lm.json', 'w') as f:
		json.dump(examples, f, indent=2)

