# Generation model for [Logic2Text](https://github.com/czyssrs/Logic2Text)

## Data
In the dataset folder, we have the full dataset (all_data.json), and the train test split (train.json, valid.json, test.json). 
Each example is in a dictionary of the following format:
```
    "topic": table caption,
    "wiki": link to the wikipedia page,
    "action": logic type,
    "sent": description,
    "annotation": raw annotation data from the mturk,
    "logic": logic form,
    "logic_str": linearized logic form,
    "nid": number of fuction nodes,
    "table_header": table header, in list format
    "table_cont": table content, list of lists
  
```

## Logic form execution
In the execution folder, run
```
python execute.py
```
It will execute all the logic forms in all_data.json. All the function definitions are in APIs.py

This site is under construction, and we will release other codes in the future.

# Template-GPT2-Logic2Text
Go to the data/ folder, link the [all_csv](https://github.com/wenhuchen/Table-Fact-Checking/tree/master/data/all_csv) to this place:
```
cd data/
ln -s [you_all_csv_folder] .
cd ../
```
In the parent folder, run the following command to train the model
```
CUDA_VISIBLE_DEVICES=0 python GPT2.py --do_train
```
After training, run the following command to test the model
```
CUDA_VISIBLE_DEVICES=0 python GPT2-coarse-to-fine.py --do_test --load_from [YOUR_MODEL]
```
The results will be saved to the output folder (you need to create one if not exist).
