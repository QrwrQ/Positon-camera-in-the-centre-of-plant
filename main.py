import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
train=pd.read_csv("../train.csv")
test=pd.read_csv("../test.csv")

train=train.dropna(axis=0,subset=['Age','Embarked'])
test=test.dropna(axis=0,subset=['Age','Embarked'])

# print(train.isnull().sum())
input_x=train.loc[:,['Pclass', 'Sex', 'Age', 'SibSp','Parch','Fare','Embarked']]
test_x=test.loc[:,['Pclass', 'Sex', 'Age', 'SibSp','Parch','Fare','Embarked']]
print(input_x.dtypes)
output_y=train.iloc[:,1]
test_y=test.iloc[:,1]

input_x.replace('male',1,inplace=True)
input_x.replace('female',0,inplace=True)

test_x.replace('male',1,inplace=True)
test_x.replace('female',0,inplace=True)
for ele,row in input_x.iterrows():
    # print(input_x.loc[ele])
    if(type(input_x.loc[ele,'Embarked'])==str):
        input_x.loc[ele,'Embarked']=ord(input_x.loc[ele,'Embarked'])

for ele,row in test_x.iterrows():
    # print(input_x.loc[ele])
    if(type(test_x.loc[ele,'Embarked'])==str):
        test_x.loc[ele,'Embarked']=ord(test_x.loc[ele,'Embarked'])
input_x['Embarked']=input_x['Embarked'].astype('float')
test_x['Embarked']=test_x['Embarked'].astype('float')
print(input_x.dtypes)
input_x=input_x.values
test_x=test_x.values

output_y=output_y.values
test_y=test_y.values
# input_x=np.array(input_x)
X_train=torch.FloatTensor(input_x)
Y_train=torch.LongTensor(output_y)

X_test=torch.FloatTensor(test_x)
Y_test=torch.LongTensor(test_y)
# Y_train=Y_train.unsqueeze(1)
print(Y_train.shape)

# print(input_x)
class ANN_model(nn.Module):
    def __init__(self,input_feature=7,hidden1=20,hindden2=20,out_featrues=2):
        super().__init__()
        self.f_connected1=nn.Linear(input_feature,hidden1)
        self.f_connected2=nn.Linear(hidden1,hindden2)
        self.out=nn.Linear(hindden2,out_featrues)
    def forward(self,x):
        x=F.relu(self.f_connected1(x))
        x=F.relu(self.f_connected2(x))
        x=self.out(x)
        return x
torch.manual_seed(20)
model=ANN_model()
loss_function=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
epochss=7000
final_losses=[]
for i in range(epochss):
    i=i+1
    y_pred=model.forward(X_train)
    loss=loss_function(y_pred,Y_train)
    final_losses.append(loss)
    if i%10==1:
        print("Epoch number: {} and the loss : {}".format(i,loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

prediction=[]
with torch.no_grad():
    for i,data in enumerate(X_test):
        y_pred=model(data)
        prediction.append(y_pred.argmax().item())
score=accuracy_score(test_y,prediction)
print(score)