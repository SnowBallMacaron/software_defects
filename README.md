# software_defects
Required pkgs: sklearn, torch, pandas, numpy
xdm 冲冲冲

# NN method 模型相关：

## 总模型架构
data in -> Layer xN -> Linear -> Sigmoid -> out
### 总模型可更改的地方

- Layer 数量
  - 于`layer_params`中修改
- feature 数量
  - 在`preprocess_data`中修改完后更改NNModel.py中的`n_feature`
- 最后激活层函数
  - 供参考的有： 
    -  Sigmoid
    -  tanh
    -  Softmax
    -  ReLu
  -  于 `NNModel` class `__init__`中，`for`循环外修改
- （可能）使用其他非MLP架构


## Layer 架构：
Linear -> ReLU -> Dropout

### Layer可更改的地方

- Linear 神经元（参数）个数
  - 于`layer_params`中修改
- 激活层函数
  - 于 `NNModel` class `__init__`中，`for`循环内修改
- Dropout 概率

# NN method 训练相关：

### 目前训练逻辑

> 目前实际上并非正式训练，而是在做Cross Validation (CV)，但是和正常训练是一样的逻辑。

1. `modelCVer = ModelCrossValidator()` 创建一个CV任务
2. `modelCVer.validateModel()`进行CV训练
   1. 按照 `model_features` 中给的模型参数顺序，读取一个模型的参数
   2. 按顺序读取sample data，分成kfold
   3. 对于每个fold，根据当前读取的参数创建一个新的模型
   4. 训练`n_epochs`轮
   5. validate这一个fold下模型的auc，此时一个fold的任务已经完成，进入下一个fold
   6. 当k个fold都完成后，一个sample的任务就完成了
   7. 完成所有sample的任务，读取下一个模型参数，直到所有模型都完成CV
3. `modelCVer.reportCV()` 输出CV的数据

### 可更改的参数

- 每一个模型训练轮数
  - 可在`if __name__ == '__main__':`下的`n_epochs`中直接更改
  - 也可在`modelCVer = ModelCrossValidator()`中更改对应模型的`n_epochs`
- Kfold分批数量
  - `k`，等同于将数据等分k份
- Learning Rate
  - `trainNNModel`函数中`optimizer`的`lr`参数，
- Batchsize 
  - 在`validateModel`下`train_dl` 和 `val_dl`
  - 越大训练越快，但有可能会对模型的精确度略有影响（目前看来很小）
