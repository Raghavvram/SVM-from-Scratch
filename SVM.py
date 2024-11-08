import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import plotly.express as px

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(n_features)
        self.b = 0
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
    
    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)
    
    def get_params(self, deep=True):
        return {"learning_rate": self.lr, "lambda_param": self.lambda_param, "n_iters": self.n_iters}
    
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

# Load the dataset
data = pd.read_csv('diabetes.csv')
X = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].values
y = data['Outcome'].values

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Add polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=123)

# Hyperparameter tuning with cross-validation
params = {
    'learning_rate': [0.001, 0.01, 0.1],
    'lambda_param': [0.01, 0.1, 1],
    'n_iters': [1000, 2000, 3000]
}
best_accuracy = 0
best_params = None

for lr in params['learning_rate']:
    for lp in params['lambda_param']:
        for n_iter in params['n_iters']:
            svm = SVM(learning_rate=lr, lambda_param=lp, n_iters=n_iter)
            scores = cross_val_score(svm, X_train, y_train, cv=5, scoring='accuracy')
            mean_score = np.mean(scores)
            if mean_score > best_accuracy:
                best_accuracy = mean_score
                best_params = {'learning_rate': lr, 'lambda_param': lp, 'n_iters': n_iter}

clf = SVM(learning_rate=best_params['learning_rate'], lambda_param=best_params['lambda_param'], n_iters=best_params['n_iters'])
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("SVM classification accuracy:", accuracy)

# Interactive visualization with Plotly
fig = px.scatter_matrix(data, dimensions=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'], color='Outcome')
fig.update_layout(title='Scatter matrix of features', height=1000, width=1200)
fig.show()

# Function to calculate hyperplane
def get_hyperplane_value(x, w, b, offset):
    return (-w[0] * x + b + offset) / w[1]

# Interactive visualization with hyperplane
fig = go.Figure()
fig.add_trace(go.Scatter(x=X_test[:, 0], y=X_test[:, 1], mode='markers', marker=dict(color=predictions, colorscale='Viridis', size=10), name='Test Data'))

x0_1 = np.amin(X[:, 0])
x0_2 = np.amax(X[:, 0])
x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)
x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)
x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

fig.add_trace(go.Scatter(x=[x0_1, x0_2], y=[x1_1, x1_2], mode='lines', name='Hyperplane', line=dict(color='Blue', dash='dash')))
fig.add_trace(go.Scatter(x=[x0_1, x0_2], y=[x1_1_m, x1_2_m], mode='lines', name='Support Vector -', line=dict(color='Black')))
fig.add_trace(go.Scatter(x=[x0_1, x0_2], y=[x1_1_p, x1_2_p], mode='lines', name='Support Vector +', line=dict(color='Black')))

fig.update_layout(title='SVM Decision Boundary', xaxis_title='Feature 1', yaxis_title='Feature 2', height=1000, width=1200)
fig.show()
