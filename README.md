---

# **1D Linear Regression**

![image](https://github.com/user-attachments/assets/5f187b9f-3c73-49e5-8bda-a2faf657b7ab)

Welcome to the **1D Linear Regression** project! 🌟 This repository demonstrates a simple yet powerful machine learning model that predicts target values based on a single-dimensional input dataset. By exploring this project, you'll gain insights into the core workings of linear regression, gradient descent, and model optimization.

---

## **Project Overview**

Linear regression is one of the most fundamental concepts in machine learning. It aims to establish a relationship between inputs and outputs by fitting a straight line to the data. In this project:
- Input features represent environmental variables (e.g., temperature, rainfall, and humidity).
- Target variables predict the yield of two crops: apples 🍎 and oranges 🍊.

---

## **Features**

- A hands-on demonstration of linear regression using PyTorch.
- Implementation of custom loss functions (Mean Squared Error).
- Manual weight and bias updates through gradient descent.
- Clear explanation of training loops and optimization steps.

---

## **Getting Started**

### **Prerequisites**
- Python 3.7 or above.
- PyTorch installed (`pip install torch`).
- Numpy installed (`pip install numpy`).

### **Installation**
Clone the repository:
```bash
git clone https://github.com/YourUsername/1D_Linear_Regression.git
cd 1D_Linear_Regression
```

### **Usage**
1. Run the script:
   ```bash
   python linear_regression.py
   ```
2. Observe how the loss decreases over epochs as the model learns to predict.

---

## **Code Structure**
- **Inputs and Targets**: Represented as tensors for numerical computations.
- **Model Function**: A simple linear transformation (`Wx + b`).
- **Loss Function**: Mean Squared Error (MSE) to quantify prediction error.
- **Training Loop**: Gradient descent to iteratively minimize the loss.

---

## **Results**
After training for 1000 epochs, the model effectively predicts the target values for apples and oranges:
- **Final Loss**: <0.0002> 
- Predicted outputs closely match the target outputs.

---

## **Key Learnings**
- Linear regression provides a foundational understanding of machine learning.
- PyTorch simplifies tensor operations and gradient computation.
- Manual weight updates offer a deeper grasp of optimization mechanics.

---

## **Contributing**
We welcome contributions to enhance this project! Feel free to:
- Submit bug reports or feature requests.
- Fork and send pull requests.

---

## **License**
This project is licensed under the MIT License. See the LICENSE file for details.

---

## **Acknowledgments**
- Thanks to the PyTorch community for creating such an intuitive framework.
- Inspiration from real-world agricultural datasets.

---

Let me know if you'd like to tweak this or proceed with other projects! 😊
