import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import mean_squared_error, r2_score, balanced_accuracy_score, recall_score, ConfusionMatrixDisplay

# numpy 2.0 with Apple Accelerate prints false "encountered in matmul" warnings; the env var covers the parallel workers
os.environ['PYTHONWARNINGS'] = 'ignore::RuntimeWarning'
warnings.filterwarnings('ignore', category = RuntimeWarning)

df = pd.read_csv(Path(__file__).parent / 'diabetes.csv')
classes = ['Low', 'Moderate', 'High']

counts = df['Diabetes_Risk'].value_counts()[classes]
counts.plot.bar(color = ['green', 'orange', 'red'])
for i, n in enumerate(counts) :
    plt.text(i, n, f"{n} ({n / len(df) * 100:.1f}%)", ha = 'center', va = 'bottom')
plt.title('Patients by Diabetes Risk Class')
plt.xlabel('Risk Class')
plt.ylabel('Number of Patients')
plt.xticks(rotation = 0)
plt.tight_layout()
plt.show()

missing = df.isna().mean() * 100
missing[missing > 0].sort_values().plot.barh()
plt.title('Missing Values by Column')
plt.xlabel('Percent of Patients Missing')
plt.tight_layout()
plt.show()

# These columns are derived from the answer, so keeping them would leak it into the model
leaky = ['Patient_ID', 'Diabetes_Risk_Score', 'AI_Health_Recommendation', 'Doctor_Consultation_Needed']
data = df.drop(columns = leaky)

x = data.drop(columns = 'Diabetes_Risk')
y = data['Diabetes_Risk']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify = y)
x_train, x_test = x_train.copy(), x_test.copy()

num_cols = x.select_dtypes('number').columns
cat_cols = x.select_dtypes(exclude = 'number').columns

for col in x_train.columns[x_train.isna().any()] :
    x_train[col + '_missing'] = x_train[col].isna().astype(int)
    x_test[col + '_missing'] = x_test[col].isna().astype(int)

# Fill values come from the training set only
medians = x_train[num_cols].median()
modes = x_train[cat_cols].mode().iloc[0]
x_train[num_cols] = x_train[num_cols].fillna(medians)
x_test[num_cols] = x_test[num_cols].fillna(medians)
x_train[cat_cols] = x_train[cat_cols].fillna(modes)
x_test[cat_cols] = x_test[cat_cols].fillna(modes)

x_train = pd.get_dummies(x_train, drop_first = True, dtype = int)
x_test = pd.get_dummies(x_test, drop_first = True, dtype = int).reindex(columns = x_train.columns, fill_value = 0)

risk_map = {'Low' : 0, 'Moderate' : 1, 'High' : 2}
ols = LinearRegression()
ols.fit(x_train, y_train.map(risk_map))
ols_pred = ols.predict(x_test)
print(f"OLS test R^2: {r2_score(y_test.map(risk_map), ols_pred):.4f}, MSE: {mean_squared_error(y_test.map(risk_map), ols_pred):.4f}")

cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
results = []

def evaluate(name, y_pred, plot = True) :
    recalls = recall_score(y_test, y_pred, labels = classes, average = None, zero_division = 0)
    results.append({'Model' : name,
                    'Accuracy' : (y_pred == y_test).mean(),
                    'Balanced accuracy' : balanced_accuracy_score(y_test, y_pred),
                    'Recall Low' : recalls[0], 'Recall Moderate' : recalls[1], 'Recall High' : recalls[2]})
    if plot :
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels = classes)
        plt.title(f"{name}: Confusion Matrix")
        plt.tight_layout()
        plt.show()

def plot_cv(search, param, values, name, xlabel, log = False) :
    plt.plot(values, search.cv_results_['mean_test_score'], marker = 'o')
    plt.axvline(search.best_params_[param], color = 'red', linestyle = '--', label = f"Best = {search.best_params_[param]}")
    if log :
        plt.xscale('log')
    plt.title(f"{name}: CV Balanced Accuracy")
    plt.xlabel(xlabel)
    plt.ylabel('Mean CV Balanced Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

evaluate('Always High (baseline)', np.full(len(y_test), 'High'), plot = False)

# C is the inverse of the penalty strength
c_values = [0.001, 0.01, 0.1, 1, 10, 100]
logit = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(class_weight = 'balanced', max_iter = 2000)),
                     {'logisticregression__C' : c_values},
                     scoring = 'balanced_accuracy', cv = cv, n_jobs = -1)
logit.fit(x_train, y_train)
plot_cv(logit, 'logisticregression__C', c_values, 'Logistic Regression', 'C', log = True)
evaluate(f"Logistic Regression (C = {logit.best_params_['logisticregression__C']})", logit.predict(x_test))

model = logit.best_estimator_.named_steps['logisticregression']
weights = pd.DataFrame(model.coef_.T, index = x_train.columns, columns = model.classes_)
print("\nTop predictors for High risk:")
print(weights['High'].sort_values(key = abs, ascending = False).head(10).round(3).to_string())

nb = GaussianNB()
nb.fit(x_train, y_train, sample_weight = compute_sample_weight('balanced', y_train))
evaluate('Naive Bayes', nb.predict(x_test))

k_values = list(range(1, 51, 2))
knn = GridSearchCV(make_pipeline(StandardScaler(), KNeighborsClassifier()),
                   {'kneighborsclassifier__n_neighbors' : k_values},
                   scoring = 'balanced_accuracy', cv = cv, n_jobs = -1)
knn.fit(x_train, y_train)
plot_cv(knn, 'kneighborsclassifier__n_neighbors', k_values, 'k-NN', 'k (number of neighbours)')
evaluate(f"k-NN (k = {knn.best_params_['kneighborsclassifier__n_neighbors']})", knn.predict(x_test))

print()
print(pd.DataFrame(results).round(4).to_string(index = False))
