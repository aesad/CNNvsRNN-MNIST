# 🧠 Deep Learning: CNN vs RNN for MNIST Image Classification

This project compares **Convolutional Neural Networks (CNN)** and several **Recurrent Neural Network (RNN)** architectures — including **LSTM**, **GRU**, and their bidirectional variants — on the **MNIST** handwritten digit dataset.  
The goal is to analyze how spatial (CNN) and sequential (RNN family) models handle 2D image data.

---

## 📁 Project Overview

**Notebook:** [`CNNvsRNN-MNIST.ipynb`](./CNNvsRNN_MNIST.ipynb)

**Dataset:** [MNIST Handwritten Digits](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

**Framework:** PyTorch

---

## ⚙️ 1. Project Stages

### **1️⃣ Data Preparation**
- MNIST images (28×28 grayscale) are normalized using the dataset mean and standard deviation:  
  ```python
  transforms.Normalize((0.1307,), (0.3081,))
  ```
- Train/test splits and DataLoaders created for efficient batching.

### **2️⃣ Model Architectures**
Implemented models:
- `CNN`: Two convolutional layers + dropout + fully connected head  
- `RNN`: Vanilla RNN treating each image row (28 pixels) as a time step  
- `LSTM`: Long Short-Term Memory network to capture longer dependencies  
- `GRU`: Gated Recurrent Unit for computational efficiency  
- `BiRNN`, `BiLSTM`, `BiGRU`: Bidirectional variants aggregating forward and backward sequence information  

Each model ends with a fully connected layer + `log_softmax` output.

### **3️⃣ Training Setup**
```python
batch_size = 64
learning_rate = 0.001
num_epochs = 5
optimizer = Adam
criterion = CrossEntropyLoss
device = "cuda" if available else "cpu"
```

### **4️⃣ Evaluation Metrics**
- **Accuracy** and **Loss** on the train and test set  
- **Training Time** (in **minutes:seconds**, MM:SS) measured for performance comparison

---

## 📊 2. Results

| Model  | Accuracy | Loss   | Time  |
|:--------|:--------:|:------:|:------:|
| **CNN**   | **0.9927** | 0.0253 | 08:46 |
| RNN   | 0.9610 | 0.1577 | 02:48 |
| LSTM  | 0.9839 | 0.0558 | 04:53 |
| GRU   | 0.9863 | 0.0492 | 04:31 |
| BiRNN | 0.9671 | 0.1149 | 04:08 |
| BiLSTM| 0.9870 | 0.0429 | 09:27 |
| BiGRU | 0.9885 | **0.0369** | 07:46 |


---

## 📈 3. Visualization

Each model’s training and validation performance is plotted using the provided `plot_history()` function.

```python
plot_history(history, model_name="CNN")
```

Example output:

- **Left:** Loss over epochs  
- **Right:** Accuracy over epochs  
- Automatically saved as `<model_name>_training_plot.png`

---
### 3.1 Visualization of CNN

<p align="center">
  <img src="./training_plot/CNN_training_plot.png" alt="CNN Training" width="1200">
</p>

**Observation:**  
- CNN achieves highest accuracy among all models.  
- Both training and validation curves closely follow each other, showing minimal overfitting.  

---

### 3.2 Visualization of RNN

<p align="center">
  <img src="./training_plot/RNN_training_plot.png" alt="RNN Training" width="1200">
</p>

**Observation:**  
- Training converges faster but accuracy saturates early.
  
---

### 3.3 Visualization of LSTM

<p align="center">
  <img src="./training_plot/LSTM_training_plot.png" alt="LSTM Training" width="1200">
</p>

**Observation:**  
- LSTM improves over vanilla RNN.  
- Validation accuracy nearly matches CNN performance, showing stronger sequence modeling.  

---

### 3.4 Visualization of GRU

<p align="center">
  <img src="./training_plot/GRU_training_plot.png" alt="GRU Training" width="1200">
</p>

**Observation:**  
- GRU achieves slightly superior accuracy to the LSTM with a faster training time..  

---

### 3.5 Visualization of BiRNN

<p align="center">
  <img src="./training_plot/BiRNN_training_plot.png" alt="BiRNN Training" width="1200">
</p>

**Observation:**  
- Bidirectionality slightly improves test accuracy but doubles the computational cost.  

---

### 3.6 Visualization of BiLSTM

<p align="center">
  <img src="./training_plot/BiLSTM_training_plot.png" alt="BiLSTM Training" width="1200">
</p>

**Observation:**  
- BiLSTM achieves strong results comparable to GRU and CNN.  
- Both directions capture sequence information, enhancing recognition accuracy.  

---

### 3.7 Visualization of BiGRU

<p align="center">
  <img src="./training_plot/BiGRU_training_plot.png" alt="BiGRU Training" width="1200">
</p>

**Observation:**  
- BiGRU shows excellent stability and efficiency, ranking second after CNN.  
- It combines fast training with strong bidirectional context learning.  

## 💬 4. Discussion

### **CNN:**
- Achieved the **highest accuracy (99.27%)** and **lowest loss** among all models.  
- CNNs are best for processing spatial or grid-like data like images, by using filters to identify local patterns.
- However, training time is relatively longer due to convolutional operations.

### **RNN and Bidirectional RNN:**
- Performed **significantly worse** (96.1–96.7%) because vanilla RNNs are designed for sequential data like text or time series, by using internal memory to process data in order and understand temporal dependencies and are not ideal for 2D spatial patterns.
- Bidirectional RNN improved slightly but at a cost in training time.

### **LSTM and GRU Families:**
- **LSTM** and **GRU** achieved strong results (~98.6–98.9%) — better than simple RNNs but below CNN.
- **BiLSTM** and **BiGRU** further improved performance by processing sequences in both directions.
- **BiGRU**  reached a good balance between accuracy and speed, with slightly faster training than BiLSTM.

---

## 💡 5. Possible Improvements

Regarding recurrent neural networks (RNNs), our study has been limited to conventional models. However, numerous advanced and contemporary RNN architectures have demonstrated strong performance in text processing tasks. Investigating their effectiveness in image processing is a promising area of study. For instance, the [**Recurrent Attention Unit (RAU)**](./[CNNvsRNN_MNIST.ipynb](https://www.sciencedirect.com/science/article/abs/pii/S0925231222013339)) enhances a Gated Recurrent Unit (GRU) cell by incorporating an attention gate, thereby seamlessly integrating the **attention mechanism** within the cell’s structure.

---

## 🧩 6. Repository Structure

```
📂 CNN-vs-RNN-MNIST
├── 📁 models
│   ├── CNN.py
│   ├── RNN.py
│   ├── LSTM.py
│   └── GRU.py
│
├── 📁 training_plot
│   ├── CNN_training_plot.png
│   ├── RNN_training_plot.png
│   ├── LSTM_training_plot.png
│   ├── GRU_training_plot.png
│   ├── BiRNN_training_plot.png
│   ├── BiLSTM_training_plot.png
│   └── BiGRU_training_plot.png
│
├── 📘 CNNvsRNN_MNIST.ipynb
└── 📄 README.md

```

---

## 📚 References

- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-based learning applied to document recognition.*
- Hochreiter, S., & Schmidhuber, J. (1997). *Long short-term memory.*
- Cho, K. et al. (2014). *Learning phrase representations using RNN encoder–decoder for statistical machine translation.*
