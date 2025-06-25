from preprocess import prepare_dataset
from neural_network import NeuralNetwork
from utils import preprocess_single_image
from sklearn.model_selection import train_test_split
import os

#dataset
dataset_path=os.path.join(os.path.join(os.getcwd(),'dataset'))
X,y=prepare_dataset(dataset_path)

#split
X_train,X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#initialize the nn
nn=NeuralNetwork()

#train nn
nn.train(X_train,y_train,epochs=1000,batch_size=128)

#train
train_preds=nn.predict(X_train)
train_accuracy=(train_preds==y_train).mean()
print(f"Training Accuracy:{train_accuracy*100:.2f}%")

#test
test_preds = nn.predict(X_test)
test_accuracy = (test_preds == y_test).mean()
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

#real image
test_image_path = os.path.join(dataset_path, 'real', 'original_1_12.png')
test_image = preprocess_single_image(test_image_path)
prediction = nn.predict(test_image)

if prediction[0][0]==1:
    print("Prediction: real signature")
else:
    print("Prediction: fake signature")
    