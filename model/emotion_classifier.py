import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


class TextPreprocessor:
    def __init__(self):
        # Download all required NLTK data
        nltk_resources = [
            'punkt',
            'wordnet',
            'stopwords',
            'averaged_perceptron_tagger',
            'punkt_tab'
        ]
        for resource in nltk_resources:
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                print(f"Warning: Could not download {resource}: {str(e)}")
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"Warning: Error initializing lemmatizer or stopwords: {str(e)}")
            self.lemmatizer = None
            self.stop_words = set()

    def preprocess_text(self, text):
        """Preprocess single text input"""
        try:
            # Handle NaN or non-string input
            if pd.isna(text) or not isinstance(text, str):
                return ""

            # Convert to lowercase
            text = text.lower()

            # Remove special characters and numbers
            text = re.sub(r'[^a-zA-Z\s]', '', text)

            # Simple word splitting if NLTK tokenization fails
            try:
                tokens = word_tokenize(text)
            except:
                tokens = text.split()

            # Lemmatize and remove stopwords if possible
            if self.lemmatizer and self.stop_words:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens
                          if token not in self.stop_words and len(token) > 2]
            else:
                tokens = [token for token in tokens if len(token) > 2]
            return ' '.join(tokens)
        except Exception as e:
            print(f"Warning: Error in preprocessing text: {str(e)}")
            return text


class EmotionClassifier:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.class_names = None

    def load_and_preprocess_data(self, train_path, test_path, val_path):
        """Load and preprocess the datasets"""
        # Load data
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        val_df = pd.read_csv(val_path)

        # Preprocess text
        print("Preprocessing text data...")
        for df in [train_df, test_df, val_df]:
            df['processed_text'] = df['text'].apply(self.preprocessor.preprocess_text)

        # Encode labels
        self.label_encoder.fit(train_df['label'])

        X_train = train_df['processed_text']
        y_train = self.label_encoder.transform(train_df['label'])
        X_test = test_df['processed_text']
        y_test = self.label_encoder.transform(test_df['label'])
        X_val = val_df['processed_text']
        y_val = self.label_encoder.transform(val_df['label'])

        self.class_names = list(self.label_encoder.classes_)
        return X_train, y_train, X_test, y_test, X_val, y_val

    def create_pipeline(self):
        """Create the model pipeline with SMOTE"""
        return ImbPipeline([
            ('tfidf', TfidfVectorizer(
                analyzer='word',
                token_pattern=r'\w{1,}',
                max_features=5000,
                lowercase=True
            )),
            ('smote', SMOTE(random_state=42)),
            ('classifier', LogisticRegression(
                max_iter=2000,
                solver='lbfgs',
                penalty='l2',
                class_weight='balanced',
                random_state=42
            ))
        ])

    def train_model(self, train_path, test_path, val_path):
        """Train the model with grid search and SMOTE"""
        # Load and preprocess data
        X_train, y_train, X_test, y_test, X_val, y_val = self.load_and_preprocess_data(
            train_path, test_path, val_path
        )

        # Create pipeline
        pipeline = self.create_pipeline()

        # Define parameter grid
        param_grid = {
            'tfidf__max_features': [5000, 7000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__min_df': [2, 3],
            'classifier__C': [0.1, 1.0, 10.0]
        }

        # Perform grid search with stratification
        print("Starting hyperparameter tuning...")
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1_macro',
            n_jobs=-1,
            verbose=2,
            error_score='raise'
        )

        # Fit the model
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_

        # Print best parameters
        print("\nBest parameters found:")
        print(grid_search.best_params_)

        # Evaluate and plot results
        self.evaluate_model(X_val, y_val, "Validation")
        self.evaluate_model(X_test, y_test, "Test")

        # Save the model
        self.save_model('emotion_classifier.joblib')

    def evaluate_model(self, X, y, dataset_name):
        """Evaluate model performance and plot confusion matrix"""
        predictions = self.model.predict(X)

        print(f"\n{dataset_name} Set Results:")
        try:
            print(classification_report(y, predictions, target_names=self.class_names, zero_division=0))
        except Exception as e:
            print("Error generating classification report:", str(e))
            print("\nFallback to basic metrics:")
            from sklearn.metrics import accuracy_score, f1_score
            print(f"Accuracy: {accuracy_score(y, predictions):.4f}")
            print(f"Macro F1: {f1_score(y, predictions, average='macro'):.4f}")

        # Plot confusion matrix
        try:
            cm = confusion_matrix(y, predictions)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.class_names,
                        yticklabels=self.class_names)
            plt.title(f'Confusion Matrix - {dataset_name} Set')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(f'confusion_matrix_{dataset_name.lower()}.png')
            plt.close()
        except Exception as e:
            print(f"Error generating confusion matrix plot: {str(e)}")

    def save_model(self, filename):
        """Save the model and necessary components"""
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'preprocessor': self.preprocessor,
            'class_names': self.class_names
        }
        joblib.dump(model_data, filename)

    def load_model(self, filename):
        """Load the saved model and components"""
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.preprocessor = model_data['preprocessor']
        self.class_names = model_data['class_names']

    def predict_emotion(self, text):
        """Predict emotion for new text"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        # Preprocess the input text
        processed_text = self.preprocessor.preprocess_text(text)

        # Get prediction and probability
        prediction = self.model.predict([processed_text])[0]
        probabilities = self.model.predict_proba([processed_text])[0]

        # Convert numpy types to native Python types
        prediction = int(prediction)  # Convert from np.int64 to int
        probabilities = [float(p) for p in probabilities]  # Convert from np.float64 to float

        # Get emotion and confidence - ensure we convert from numpy types to native Python types
        emotion = str(self.label_encoder.inverse_transform([prediction])[0])  # Convert to string
        confidence = float(probabilities[prediction])  # Convert to native Python float

        # Get top 3 emotions with probabilities
        top_3_idx = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)[:3]
        top_3_emotions = [
            {
                'emotion': str(self.label_encoder.inverse_transform([idx])[0]),  # Convert to string
                'confidence': float(probabilities[idx])  # Convert to float
            }
            for idx in top_3_idx
        ]

        classifier_result = {
            'emotion': emotion,
            'confidence': confidence,
            'top_3_predictions': top_3_emotions
        }
        emotion_map = {
            "0": "sadness",
            "1": 'joy',
            "2": 'love',
            "3": 'anger',
            "4": 'fear'
        }
        classifier_result['emotion'] = emotion_map[classifier_result['emotion']]
        for p in classifier_result['top_3_predictions']:
            p['emotion'] = emotion_map[p['emotion']]
        return classifier_result

if __name__ == "__main__":
    # Example usage

    # For training a new model:
    '''
    classifier = EmotionClassifier()
    classifier.train_model('training.csv', 'test.csv', 'validation.csv')
    '''
