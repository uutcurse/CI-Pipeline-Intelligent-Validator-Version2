from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import joblib

class TFIDFBaseline:
    def __init__(self, vectorizer_params, model_params):
        self.vectorizer_params = vectorizer_params
        if "ngram_range" in self.vectorizer_params and isinstance(self.vectorizer_params["ngram_range"], list):
            self.vectorizer_params["ngram_range"] = tuple(self.vectorizer_params["ngram_range"])
            
        self.model_params = model_params
        
        self.vectorizer = TfidfVectorizer(**self.vectorizer_params)
        
        m_type = self.model_params.pop("model_type")
        if m_type == "LogisticRegression":
            self.model = LogisticRegression(**self.model_params)
        elif m_type == "LinearSVC":
            self.model = LinearSVC(**self.model_params)
        else:
            raise ValueError(f"Unknown model type: {m_type}")
            
    def fit(self, X_train, y_train):
        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_vec, y_train)
        return X_train_vec
        
    def transform(self, X):
        return self.vectorizer.transform(X)
        
    def predict(self, X):
        X_vec = self.transform(X)
        return self.model.predict(X_vec)
        
    def predict_proba(self, X):
        if not hasattr(self.model, "predict_proba"):
            raise NotImplementedError("This model does not support probability predictions.")
        X_vec = self.transform(X)
        return self.model.predict_proba(X_vec)
        
    def save(self, filepath):
        joblib.dump({"vectorizer": self.vectorizer, "model": self.model}, filepath)
        
    @classmethod
    def load(cls, filepath):
        data = joblib.load(filepath)
        instance = cls({}, {"model_type": "LogisticRegression"}) # dummy params
        instance.vectorizer = data["vectorizer"]
        instance.model = data["model"]
        return instance
