import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

weights = np.zeros(2)
bias = 0
learning_rate = 0.1
epochs = 15

for epoch in range(epochs):
    for i in range(len(X)):
        weighted_sum = np.dot(X[i],weights) + bias
        if weighted_sum >= 0:
            output = 1
        else:
            output = 0
        error = y[i] - output
        weights = weights + learning_rate * error * X[i]
        bias = bias + learning_rate * error

print("Final Weights:",weights)
print("Final Bias:",bias)

test_data = np.array([1,1])
result = np.dot(test_data,weights) + bias

if result >= 0:
    prediction = 1
else:
    prediction = 0

print("Prediction for test data:",prediction)