# factors_and_price_label_prediction_model

参数、结果等详细内容见总结飞书文档

## 一、可转债ETF
1. 数据清洗&处理
1. LSTM model
   - LSTM orginal: 输出直接为fc(output)
   - LSTM with_mapping_layer
   - ALSTM - LSTM with temporal attention layer
   - Adv-ALSTM - 加入对抗的ALSTM模型
   - Adv-ALSTM with trained weight matrix - 加注意力层时train了构造a的W matrix

## 二、REITs因子
1. 因子构造
   - 宏观
   - 量价
1. 因子筛选
   - 量价IC筛选
   - 共线性
   - 遗传算法
   - 回归
   - lightGBM

## Contact
xieanranjudy@gmail.com
