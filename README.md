# Iris Classification - PyTorch (Assessment 1)

##  Objective
Build and train a basic 2-layer neural network in PyTorch to classify Iris flowers into 3 categories using the Iris dataset.

---

##  Project Contents
- `iris_classification.ipynb`
- Accuracy-vs-Epoch plot
- User input prediction for testing the model manually

---

##  How to Run

1. Open the notebook in Jupyter or Google Colab.
2. Run the cells one by one to:
   - Load and preprocess the Iris dataset
   - Train a 2-layer neural network
   - Plot training and testing accuracy
   - Test the model with custom input

---

##  Model Summary

- Input: 4 features (sepal & petal size)
- Hidden layer: 10 neurons + ReLU
- Output: 3 classes (Setosa, Versicolor, Virginica)
- Optimizer: SGD
- Loss: CrossEntropyLoss
- Epochs: 50

---

##  How to Predict

At the end of the notebook, you can enter custom input like this:

```
Sepal Length (cm): 5.1  
Sepal Width (cm): 3.5  
Petal Length (cm): 1.4  
Petal Width (cm): 0.2  
```

Output:
```
 Predicted Flower: Setosa
```

---

##  Result

- Training Accuracy: ~78%
- Testing Accuracy: ~80–90%
- Accuracy improves clearly over 50 epochs

---

## Requirements

No installation needed if using Google Colab.  
If running locally, install:

```bash
pip install torch matplotlib scikit-learn pandas
```

---

##  Learnings
- How to train a basic neural network in PyTorch from scratch
- Train-test split and normalization
- Tracking and plotting accuracy
- Real-time prediction with user input
