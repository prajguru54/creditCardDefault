import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

class ModelFinder:

    def __init__(self, x, y) -> None:
        self.x, self.y = x, y

    def find_best_model(self, estimators, param_grids, cv = 5):
        self.estimators = estimators
        self.param_grids = param_grids
        self.cv = cv
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size = 0.2)
        results = []
        for estimator, params in zip(self.estimators, self.param_grids):
            gcv = GridSearchCV(estimator, param_grid=params, cv = self.cv, scoring='f1', verbose=True)
            gcv.fit(self.x_train, self.y_train)
            self.score = gcv.score(self.x_test, self.y_test)
            results.append((gcv.best_estimator_, self.score))
        return results
        

    def plot_model_perfomence(self, results):
        self.model_names = [str(results[i][0].__class__).split(".")[-1].strip("'>") for i in range(len(results))]
        self.scores = [results[i][1] for i in range(len(results))]
        plt.figure(figsize=(10,5))
        plt.bar(self.model_names, self.scores, width=0.4)
        plt.title("Perfomence of various models")
        plt.xlabel("Model")
        plt.ylabel("f1 Scores")
        plt.savefig('Plots/model_perfomence.png')
        
