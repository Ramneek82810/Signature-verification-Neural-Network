# ✍️ Signature Verification Neural Network

A deep learning-based signature verification system that classifies and authenticates signatures as genuine or forged using Convolutional Neural Networks (CNN). This project demonstrates a practical application of neural networks in biometric authentication.

## 🚀 Features

- 🧠 Uses CNN for feature extraction and classification  
- 🖼️ Processes image data for both genuine and forged signatures  
- 📁 Supports structured dataset loading  
- 📊 Evaluates model accuracy on training and test sets  
- ✅ Simple and extendable architecture  

## 📁 Project Structure

```
Signature-verification-Neural-Network/
│
├── dataset/                # Contains genuine and forged signature samples
│
├── main.py                # Defines the CNN architecture
├── train.py                # Training loop and data preprocessing
├── test.py                 # Model evaluation script
└── README.md               # You're reading it!
```

## ⚙️ Installation

### 1. Clone the Repository

```
git clone https://github.com/Ramneek82810/Signature-verification-Neural-Network.git
cd Signature-verification-Neural-Network
```

### 2. Create and Activate Virtual Environment (optional but recommended)

```
python -m venv .venv
source .venv/bin/activate    # On Windows use: .venv\Scripts\activate
```

### 3. Install Dependencies

```
pip install tensorflow keras numpy matplotlib opencv-python
```

## 🧪 Run the Project

### Train the Model:

```
python train.py
```

### Test the Model:

```
python test.py
```

## 💡 How It Works

- Preprocesses images (resizing, grayscale conversion, normalization)  
- Trains a Convolutional Neural Network on labeled signature data  
- Evaluates model performance in identifying genuine vs. forged signatures  
- Can be extended with Siamese networks for one-shot verification  

## 📌 Todo

- Add GUI using Streamlit or Tkinter  
- Implement one-shot learning for unseen signatures  
- Add support for real-time camera-based signature input  
- Improve accuracy with advanced preprocessing  

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.


