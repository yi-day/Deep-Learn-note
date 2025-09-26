import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch



class TitanicDataset(Dataset):
    def __init__(self,file_path):
        self.file_path=file_path
        self.mean={
            "Pclass": 2.236695,
            "Age": 29.699118,
            "SibSp": 0.512605,
            "Parch": 0.431373,
            "Fare": 34.694514,
            "Sex_female": 0.365546,
            "Sex_male": 0.634454,
            "Embarked_C": 0.182073,
            "Embarked_Q": 0.039216,
            "Embarked_S": 0.775910
        }

        self.std = {
            "Pclass": 0.838250,
            "Age": 14.526497,
            "SibSp": 0.929783,
            "Parch": 0.853289,
            "Fare": 52.918930,
            "Sex_female": 0.481921,
            "Sex_male": 0.481921,
            "Embarked_C": 0.386175,
            "Embarked_Q": 0.194244,
            "Embarked_S": 0.417274
        }
        self.data=self._load_data()
        # 计算特征维度（总列数减去标签列）
        self.feature_size=len(self.data.columns)-1
    def _load_data(self):
        df=pd.read_csv(self.file_path)
        df=df.drop(columns=['PassengerId',"Name","Ticket","Cabin"])
        df=df.dropna(subset=["Age"])
        df = pd.get_dummies(df, columns=["Sex", "Embarked"], dtype=int)

        base_features=["Pclass","Age","SibSp","Parch","Fare"]
        for i in range(len(base_features)):
            df[base_features[i]]=(df[base_features[i]]-self.mean[base_features[i]])/self.std[base_features[i]]
        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        # 获取指定索引的特征和标签
        features=self.data.drop(columns=["Survived"]).iloc[idx].values
        label=self.data["Survived"].iloc[idx]
        # 将数据转换为PyTorch张量
        return torch.tensor(features,dtype=torch.float32),torch.tensor(label,dtype=torch.float32)

# 定义逻辑回归模型类
class LogisticRegressionModel(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        # nn.Linear也继承自nn.Module，输入为input_dim,输出一个值
        self.linear=nn.Linear(input_dim,1)

    def forward(self,x):
        # Logistic Regression 输出概率
        return torch.sigmoid(self.linear(x))

train_dataset=TitanicDataset("../data/train.csv")
validation_dataset=TitanicDataset("../data/validation.csv")

# 创建模型实例，输入维度为训练集的特征维度
model=LogisticRegressionModel(train_dataset.feature_size)
#model.to("cuda")
# 设置模型为训练模式
model.train()

# 定义优化器（随机梯度下降）
optimizer=torch.optim.SGD(model.parameters(),lr=0.1)

epochs=100

for epoch in range(epochs):
    correct=0 # 记录正确预测的样本数
    step=0 # 记录训练步数
    total_loss=0 # 记录总损失
    # 使用DataLoader批量加载训练数据
    for features,labels in DataLoader(train_dataset,batch_size=256,shuffle=True):
        step+=1
        #features=features.to("cuda")
        #labels=labels.to("cuda)
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播，获取模型输出
        outputs=model(features).squeeze()# 去除多余的维度
        # 计算正确预测的数量（预测概率≥0.5视为正类）
        correct+=torch.sum(((outputs>=0.5)==labels))
        # 计算二元交叉熵损失
        loss=torch.nn.functional.binary_cross_entropy(outputs,labels)
        total_loss+=loss.item()
        loss.backward()
        # 更新模型参数
        optimizer.step()
    print(f'Epoch{epoch+1},Loss:{total_loss/step:.4f}')
    print(f"Training Accuracy:{correct/len(train_dataset)}")

    # 设置模型为评估模式
    model.eval()
    with torch.no_grad(): # 在评估时不计算梯度
        correct=0
        for features,labels in DataLoader(validation_dataset,batch_size=256):
            #features=features.to("cuda")
            #labels=labels.to("cuda")
            outputs=model(features).squeeze()
            correct+=torch.sum(((outputs>=0.5)==labels))
        print(f'Validation Accuracy:{correct/len(validation_dataset)}')
