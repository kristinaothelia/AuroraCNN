import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import seaborn as sns
import termplotlib as tpl

from datetime import datetime, date
from typing import Union
from pathlib import Path

import sklearn as sk
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

from tqdm import tqdm

from lbl.dataset import DatasetContainer
from lbl.dataset import DatasetLoader

from lbl.models.model import Model
from lbl.models.efficientnet.efficientnet import EfficientNet
from lbl.models.efficientnet.config import efficientnet_params

from lbl.preprocessing import (
    PadImage,
    RotateCircle,
    StandardizeNonZero,
    )

import warnings
warnings.filterwarnings("ignore")
# -----------------------------------------------------------------------------

# with only 2 classes (aurora/no aurora):
'''
path = "models/2class/b2/2021-10-05/best_validation/checkpoint-best.pth"
model = EfficientNet.from_name(model_name=model_name, num_classes=2, in_channels=1)

LABELS = {
    0: "aurora-less",
    1: "aurora"
}
'''

LABELS = {
    0: "aurora-less",
    1: "arc",
    2: "diffuse",
    3: "discrete",
}

model_names = ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4']

def predict(model_name, model_path, container, LABELS, save_file, test=False):

    today = date.today()
    if test:
        save_path = 'datasets/predicted/test/'+model_name[-2:]+'/'+save_file[-6]+'/'
        #save_path = Path('datasets/predicted/test/') / Path(model_name[-2:]) / Path(datetime.today().strftime('%Y-%m-%d')) #/ Path('/')
    else:
        save_path = 'datasets/predicted/'+model_name[-2:]+'/'
        #save_path = Path('datasets/predicted/') / Path(model_name[-2:]) / Path(datetime.today().strftime('%Y-%m-%d')) #/ Path('/')
    #save = save_path+save_file
    #save = save_path / Path(save_file)
    #save = os.path.join(save_path, save_file)

    img_size = efficientnet_params(model_name)['resolution']

    transforms = torchvision.transforms.Compose([
        lambda x: np.float32(x),
        lambda x: torch.from_numpy(x),
        lambda x: x.unsqueeze(0),
        lambda x: torch.nn.functional.interpolate(
                input=x.unsqueeze(0),
                size=img_size,
                mode='bilinear',
                align_corners=True,
                ).squeeze(0),
        StandardizeNonZero(),
        # PadImage(size=480),
        ])

    #model = Model(1, 4, 128)
    model = EfficientNet.from_name(model_name=model_name, num_classes=4, in_channels=1)

    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    model = model.to('cuda:3')
    model.eval()

    y_pred = list()
    y_true = list()

    with torch.no_grad():
        for entry in tqdm(container):

            if test:

                score = dict()
                img = entry.open()
                x = transforms(img)
                x = x.unsqueeze(0)
                x = x.to('cuda:3')

                pred = model(x).to('cpu')
                pred = torch.softmax(pred, dim=-1)
                prediction = torch.argmax(pred, dim=-1) # out

                # Update y_pred and y_true
                y_pred.append(prediction.numpy())

                if entry.label == LABELS[0]:
                    #y_true.append(torch.tensor([0]))
                    y_true.append(0)
                elif entry.label == LABELS[1]:
                    #y_true.append(torch.tensor([1]))
                    y_true.append(1)
                elif entry.label == LABELS[2]:
                    #y_true.append(torch.tensor([2]))
                    y_true.append(2)
                else:
                    #y_true.append(torch.tensor([3]))
                    y_true.append(3)

                for i, label_pred in enumerate(pred[0]):
                    score[LABELS[i]] = float(label_pred)

                entry.add_score(score)

            else:

                if entry.label is None:

                    score = dict()
                    img = entry.open()
                    x = transforms(img)
                    x = x.unsqueeze(0)
                    x = x.to('cuda:3')

                    pred = model(x).to('cpu')
                    pred = torch.softmax(pred, dim=-1)
                    prediction = torch.argmax(pred, dim=-1)

                    for i, label_pred in enumerate(pred[0]):
                        score[LABELS[i]] = float(label_pred)

                    entry.label = LABELS[int(prediction[0])]
                    entry.human_prediction = False
                    entry.add_score(score)

    # save file with predictions
    #container.to_json(path='./datasets/Full_aurora_predicted.json')
    #container.to_json(path=save_file)
    container.to_json(path="datasets/predicted/"+save_file)

    # additional metrics
    if test:
        print(len(y_pred))
        print(len(y_true))

        def metrics(y_true, y_pred):
            report = sk.metrics.classification_report(y_true, y_pred, target_names=['no a','arc','diff','disc'])
            f1 = f1_score(y_true, y_pred, average=None) #The best value is 1 and the worst value is 0
            accuracy =accuracy_score(y_true, y_pred)
            accuracy_w = balanced_accuracy_score(y_true, y_pred) #The best value is 1 and the worst value is 0 when adjusted=False
            CM_sk = sk.metrics.confusion_matrix(y_true, y_pred, normalize='true')

            return CM_sk, accuracy, accuracy_w, f1, report

        CM, acc, acc_w, f1, report = metrics(y_true, y_pred)
        print(report)

        Path(save_path).mkdir(parents=True, exist_ok=True)
        #if not os.path.exists(save_path):
        #    os.mkdir(save_path)
        name = os.path.join(save_path, "log.txt")
        log = open(name, "w")
        log.write(str(today))
        log.write("\nf1 score (all classes): {}\n".format(f1))
        log.write("acc (w): {}. acc:{}\n\n".format(acc_w, acc))
        log.write(report)
        log.close()

        # Normalized
        N_cm = CM/CM.sum(axis=1)[:, np.newaxis]
        class_names = [r'no aurora', r'arc', r'diffuse', r'discrete']

        plt.figure() # figsize=(15,10)
        df_cm = pd.DataFrame(N_cm, index=class_names, columns=class_names).astype(float)
        heatmap = sns.heatmap(df_cm, annot=True, fmt=".2f")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=12)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=12)
        plt.ylabel(r'Observed class',fontsize=13) # True label
        plt.xlabel(r'Predicted class',fontsize=13)
        plt.title(r'Norm. confusion matrix for EfficientNet model B{}'.format(model_name[-1])+'\n'+r'Test accuracy: {:.2f}'.format(acc),fontsize=14)
        #plt.show(block=True)
        plt.tight_layout()
        plt.savefig(str(save_path) + "CM_normalized_test.png")



def Test(model_name, model_path, LABELS, num):

    json_file = 'Full_aurora_ml_test_set.json'
    container = DatasetContainer.from_json('datasets/'+json_file)
    save_file = json_file[:-5]+'_predicted_'+model_name+'_'+str(num)+'.json'

    predict(model_name, model_path, container, LABELS, save_file, test=True)


def Predict_on_unlabeld_data(model_name, model_path, mlnodes_path, LABELS):

    json_file = 'Full_aurora_new_rt_ml.json'
    container = DatasetContainer.from_json('datasets/'+json_file)

    #save_file = mlnodes_path+json_file[:-5]+'_predicted_'+model_name+'.json'
    save_file = json_file[:-5]+'_predicted_'+model_name+'_TESTNEW_.json'

    predict(model_name, model_path, container, LABELS, save_file)

# make predictions with chosen model and data set
mlnodes_path = '/itf-fi-ml/home/koolsen/Master/'
# Load a saved model. UPDATE
model_name = model_names[3]
#model_path = "models/b2/2021-10-02/best_validation/checkpoint-best.pth"
#model_path = "models/report/b3_16/best_validation/checkpoint-best.pth"
model_path = "models/report/best_validation/checkpoint-best.pth"

"""
json_file = 'datasets/Full_aurora_new_rt_ml.json'

json_file = 'datasets/Full_aurora_test_set.json'    # TEST FILE
#container = DatasetContainer.from_json(mlnodes_path+json_file)
container = DatasetContainer.from_json(json_file)
#save_file = mlnodes_path+json_file[:-5]+'_predicted_'+model_name+'.json'
#save_file = json_file[:-5]+'_predicted_'+model_name+'_TESTNEW_.json'
save_file = json_file[:-5]+'_predicted_'+model_name+'.json'

predict(model_name, model_path, container, LABELS, save_file, test=True)
"""

#Predict_on_unlabeld_data(model_name, model_path, mlnodes_path, LABELS)
#Test(model_name, model_path, LABELS)

def Test_B3():

    model_name = model_names[3]
    #model_path = "models/report/best_validation/checkpoint-best.pth"

    # num = [0, 2, 3, 4]
    #model_path1 = "models/b3/bilinear/batch_size_24/lr_0.01/st_75/g_0.1_wFalse/2022-01-28/best_validation/checkpoint-best.pth"
    #model_path2 = "models/b3/bilinear/batch_size_24/lr_0.01/st_75/g_0.1_wTrue/2022-01-28/best_validation/checkpoint-best.pth"
    #model_path3 = "models/b3/bicubic/batch_size_24/lr_0.01/st_75/g_0.1_wFalse/2022-01-29/best_validation/checkpoint-best.pth"
    #model_path4 = "models/b3/bilinear/batch_size_16/lr_0.01/st_75/g_0.1_wFalse/2022-01-29/best_validation/checkpoint-best.pth"

    num = [5, 6, 7]
    model_path1 = "models/b3/bicubic/batch_size_24/lr_0.01/st_75/g_0.1_wFalse/2022-01-30/best_validation/checkpoint-best.pth"
    model_path2 = "models/b3/bilinear/batch_size_24/lr_0.01/st_75/g_0.1_wFalse/2022-01-29/best_validation/checkpoint-best.pth"
    model_path3 = "models/b3/bilinear/batch_size_24/lr_0.01/st_75/g_0.1_wTrue/2022-01-30/best_validation/checkpoint-best.pth"

    model_paths = [model_path1, model_path2, model_path3]

    for i in range(len(model_paths)):
        Test(model_name, model_paths[i], LABELS, num[i])

def Test_B2():

    num = 8
    model_name = model_names[2]
    model_path = "models/b2/bilinear/batch_size_32/lr_0.01/st_75/g_0.1_wFalse/2022-01-31/best_validation/checkpoint-best.pth"

    Test(model_name, model_path, LABELS, num)

Test_B3()
Test_B2()

"""
# Load json file to add predictions
json_file = 'Aurora_G_omni_mean.json'
container = DatasetContainer.from_json(mlnodes_path+json_file)
save_file = mlnodes_path+json_file[:-5]+'_predicted_'+model_name+'.json'

predict(model_name, model_path, container, LABELS, save_file)


# Load json file to add predictions
#json_file = 'Aurora_G.json'
json_file = 'Aurora_4yr_G_omni_mean.json'
container = DatasetContainer.from_json(mlnodes_path+json_file)
save_file = mlnodes_path+json_file[:-5]+'_predicted_'+model_name+'.json'

predict(model_name, model_path, container, LABELS, save_file)
"""
