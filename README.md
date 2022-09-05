# KDD CUP 2022


Please install the libraries as follows:
* python 3.7
* PaddlePaddle-gpu==2.3.0
* Pandas
* Numpy

You can finish it by :

```
pip install -r requirements.txt
```
If you want to retrain the model, please make the corresponding
changes to the hyper parameters of "prepare.py" according to your
needs first. And then :

```
python train.py
```
After training the model, you can view some information such
as hyper parameters during training and loss per epoch through
the .txt file in the "logs" folder.

If you want to make predictions, please specify the hyper parameter "path_to_test_x " and then:


```
python predict.py
```
