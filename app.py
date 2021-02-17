from flask import Flask, render_template, url_for, request
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route('/')
def home():
		return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	df = pd.read_csv("spam.csv", sep='\t', names =['label', 'message'])
	# features and labels
	df['label'] = df['label'].map({'ham': 0, 'spam': 1})
	X = df['message']
	y = df['label']

	# Extract feature with CountVectorizer

	cv = CountVectorizer()
	X = cv.fit_transform(X)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

	# Naive Bayes Classifier

	clf = MultinomialNB()
	clf.fit(X_train, y_train)
	print(clf.score(X_test, y_test))

	if request.method == 'POST':
	    message = request.form['message']
	    data = [message]
	    vect = cv.transform(data).toarray()
	    my_prediction = clf.predict(vect)
	return render_template('index.html', prediction = my_prediction) 

if __name__ == '__main__':
	app.run(debug = True)

