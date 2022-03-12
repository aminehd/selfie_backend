import pandas as pd
from pathlib import Path

import IPython.display
import numpy as np
import matplotlib as plt
from fastai.data.all import *
from fastai.vision.all import *
from scipy import stats

wd = Path(r'./data/SCUT-FBP5500_v2')
output = wd / 'myoutputs'
images_path = wd/'Images'

cm_ratings = pd.read_excel(wd/'All_Ratings.xlsx',sheet_name='Caucasian_Male') 
print(load_learner)
sorted_cm_df = cm_ratings.groupby('Filename').mean('Rating').sort_values(by=['Rating'])
cm_ratings = {}
sorted_cm_list = []
for index,row in sorted_cm_df.iterrows():
    cm_ratings[index] = row["Rating"]
    sorted_cm_list.append([index,row["Rating"]])

def get_cm_paths(path):
    return list(path.glob('CM*'))

def get_rating(path):
    return cm_ratings[path.name]

def modified_get_image(images_path):
    print(images_path)
    images_list = get_image_files(images_path)
    print([pathi for pathi in images_list if pathi.name in cm_ratings.keys()]) 
    # raise ValueError(cm_ratings[images_list[0].name])
    return [pathi for pathi in images_list ]

cm_block = DataBlock(
    blocks=(ImageBlock,RegressionBlock(n_out=1)),
    # get_items=modified_get_image,
    get_items=get_cm_paths,
    get_y=get_rating,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
)

cm_data = cm_block.dataloaders(images_path)

learn = cnn_learner(cm_data,resnet34,metrics=mae)
learn.load('./model_25')
learn.export(fname='./models/export.pkl')


learn2 = load_learner(r'./models/export.pkl')
print( learn2.predict(r'./resources/CM4.jpg'))


def predict( file_name):
    # getrating
    return learn2.predict(file_name)

        # if __name__ == '__main__': 
#     print(predict(r'./resources/CM4.jpg'))
#     print(predict(r'./resources/rate.png'))

#     print(predict(r'./resources/CM1.jpg'))
#     print(predict(r'./resources/CM11.jpg'))
