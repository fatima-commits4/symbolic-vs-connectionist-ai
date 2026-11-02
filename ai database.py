#AI paradigms comparison: connectionist vs database

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

iris = load_iris() #load the dataset
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(  #split database into training and testing
    X, y, test_size=0.3, random_state=42
)

decision_tree = DecisionTreeClassifier() #symbolic AI: Decision tree
decision_tree.fit(X_train, y_train)

dt_predictions = decision_tree.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)

neural_network = MLPClassifier( #connectionist AI: neutral network
    hidden_layer_sizes=(10,),
    max_iter=1000,
    random_state=42
)
neural_network.fit(X_train, y_train)

nn_predictions = neural_network.predict(X_test)
nn_accuracy = accuracy_score(y_test, nn_predictions)

print("Symbolic AI (Decision Tree) Accuracy:", dt_accuracy) #results
print("Connectionist AI (Neural Network) Accuracy:", nn_accuracy)
