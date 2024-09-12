import numpy as np
import pickle
import os
import time  
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm  

def load_cifar10_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    features = batch[b'data'].reshape((len(batch[b'data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch[b'labels']
    return features, labels

def load_cifar10(data_dir):
    train_data = []
    train_labels = []
    for i in range(1, 6):
        features, labels = load_cifar10_batch(os.path.join(data_dir, f'data_batch_{i}'))
        train_data.append(features)
        train_labels.extend(labels)
    
    train_data = np.concatenate(train_data)
    train_labels = np.array(train_labels)
    
    test_features, test_labels = load_cifar10_batch(os.path.join(data_dir, 'test_batch'))
    return (train_data, train_labels), (test_features, test_labels)

class SimpleCNN:
    def __init__(self, saved_model=None):
        if saved_model:
            self.conv1_filters = saved_model['conv1_filters']
            self.conv1_bias = saved_model['conv1_bias']
            self.fc1_weights = saved_model['fc1_weights']
            self.fc1_bias = saved_model['fc1_bias']
        else:
            self.conv1_filters = np.random.randn(32, 3, 5, 5) * 0.1
            self.conv1_bias = np.zeros((32, 1))
            self.fc1_weights = np.random.randn(32 * 28 * 28, 10) * 0.1
            self.fc1_bias = np.zeros((10, 1))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def conv2d(self, x, filters, bias):
        n_filters, n_channels, f_height, f_width = filters.shape
        n_samples, height, width, _ = x.shape
        
        out_height = height - f_height + 1
        out_width = width - f_width + 1
        
        output = np.zeros((n_samples, out_height, out_width, n_filters))
        
        for i in range(out_height):
            for j in range(out_width):
                x_slice = x[:, i:i+f_height, j:j+f_width, :]
                for k in range(n_filters):
                    output[:, i, j, k] = np.sum(x_slice * filters[k].T, axis=(1,2,3)) + bias[k]
        
        return output
    
    def forward(self, x):
        conv1 = self.conv2d(x, self.conv1_filters, self.conv1_bias)
        relu1 = self.relu(conv1)
        flattened = relu1.reshape(x.shape[0], -1)
        fc1 = np.dot(flattened, self.fc1_weights) + self.fc1_bias.T
        output = self.softmax(fc1.T).T
        return output
    
    def save_model(self, filename):
        model_data = {
            'conv1_filters': self.conv1_filters,
            'conv1_bias': self.conv1_bias,
            'fc1_weights': self.fc1_weights,
            'fc1_bias': self.fc1_bias
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filename}")
    
    def train(self, X, y, epochs=10, batch_size=128, learning_rate=0.01, save_path='simple_cnn_model.pkl'):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs} started...")
            
            start_time = time.time()  # Start time of the epoch

            # Use tqdm to display progress of batch iteration within each epoch
            for i in tqdm(range(0, X.shape[0], batch_size), desc=f"Training Epoch {epoch+1}", unit="batch"):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                output = self.forward(X_batch)
                
                # Clipping to avoid log(0)
                output = np.clip(output, 1e-10, 1.0)
                
                error = output - np.eye(10)[y_batch]
                
                flattened = self.relu(self.conv2d(X_batch, self.conv1_filters, self.conv1_bias)).reshape(X_batch.shape[0], -1)
                self.fc1_weights -= learning_rate * np.dot(flattened.T, error)
                self.fc1_bias -= learning_rate * np.sum(error, axis=0, keepdims=True).T
            
            # Save the model after every epoch
            self.save_model(save_path)
            
            # Calculate loss and print it at the end of the epoch
            loss = -np.log(output[range(X_batch.shape[0]), y_batch])
            end_time = time.time()  # End time of the epoch

            # Calculate time taken for the epoch
            time_taken = end_time - start_time
            print(f"Epoch {epoch+1} completed, Loss: {np.mean(loss)}")
            print(f"Time taken for epoch {epoch+1}: {time_taken:.2f} seconds")

if __name__ == "__main__":
    data_dir = 'cifar-10-batches-py'  # Replace with your actual path
    (X_train, y_train), (X_test, y_test) = load_cifar10(data_dir)

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Check if a saved model exists, and load it if it does
    model_file = 'simple_cnn_model.pkl'
    if os.path.exists(model_file):
        with open(model_file, 'rb') as f:
            saved_model = pickle.load(f)
        model = SimpleCNN(saved_model)
        print("Resuming from saved model.")
    else:
        model = SimpleCNN()

    # Train the model
    model.train(X_train, y_train, epochs=50, save_path=model_file)

    # Predict on test data
    y_pred = np.argmax(model.forward(X_test), axis=1)

    # Calculate metrics
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")