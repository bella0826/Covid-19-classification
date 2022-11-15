# Covid-19-classification
環境、語言、套件: VS code、Python、Pytorch

## 資料集

COVID-19 Radiography Database

## grad_cam_for_covid.py
為訓練出來的模型做grad cam視覺化
其中register_backward_hook失敗 故程式失敗
但prediction為正確的答案

## grad_cam.py
使用另一種方式抓取梯度
成功輸出heatmap圖

