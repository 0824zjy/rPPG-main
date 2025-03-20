python main.py --config_file ./configs/train_configs/UBFC_UBFC_PURE_PHYSNET_BASIC.yaml

1       UBFC_UBFC_PURE_EFFICIENTPHYS.yaml
1       UBFC_UBFC_PURE_DEEPPHYS_BASIC.yaml
1       UBFC_UBFC_PURE_TSCAN_BASIC.yaml
注意事项1：预处理只需要进行一次；因此，在第一次训练网络之后，当你再次训练时，请在yaml文件中关闭预处理选项。

注意事项2：示例yaml文件设置将使用数据集中80%的数据进行训练，20%的数据进行验证。训练完成后，它将使用表现最佳（验证损失最小）的模型在UBFC数据集测试集上进行测试。


UBFC数据集格式如下
UBFCData/
|   |-- subject1/
|      |-- ground_truth.txt
|      |-- vid.avi
|   |-- subject3/
|      |-- ground_truth.txt
|      |-- vid.avi
|...
|   |-- subject49/
|      |-- ground_truth.txt
|      |-- vid.avi

PURE数据集格式如以下
PURE_Data/
|   |-- 01-01/
|      |-- Image1392643993642815000.png
|      |-- ...
|      |-- Image1392644060476158000.png
|      |-- ...
|      |-- Image1392644061342837000.png
|      |-- 01-01.json
|   |-- 01-02/
|      |-- Image1392644182276178000.png
|      |-- ...
|      |-- Image1392644248642882000.png
|      |-- ...
|      |-- Image1392644249876197000.png
|      |-- 01-02.json
|...
|   |-- ii-jj/
|      |-- ...
|      |-- ii-jj.json
|   |-- video/
|      |-- 01-01.avi
|      |-- 01-02.avi
|      |-- ...
|      |-- ii-jj.avi
|      |-- json/
|           |-- 01-01.json
|           |-- 01-02.json
|           |-- ...
|           |-- 10-06.json