from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import joblib

class HybridBaseline:
    def __init__(self, vectorizer_params, model_params, use_scaler=True):
        self.vectorizer_params = vectorizer_params.copy()
        if "ngram_range" in self.vectorizer_params and isinstance(self.vectorizer_params["ngram_range"], list):
            self.vectorizer_params["ngram_range"] = tuple(self.vectorizer_params["ngram_range"])
            
        self.vectorizer = TfidfVectorizer(**self.vectorizer_params)
        
        self.use_scaler = use_scaler
        self.scaler = StandardScaler() if use_scaler else None
        
        self.model_params = model_params.copy()
        m_type = self.model_params.pop("model_type")
        
        if m_type == "LogisticRegression":
            self.model = LogisticRegression(**self.model_params)
        elif m_type == "LinearSVC":
            self.model = LinearSVC(**self.model_params)
        else:
            raise ValueError(f"Unknown model type: {m_type}")
            
    def fit(self, X_text, X_struct, y_train):
        # Fit transforms on train ONLY
        X_text_vec = self.vectorizer.fit_transform(X_text)
        
        if self.use_scaler:
            X_struct_trans = self.scaler.fit_transform(X_struct)
        else:
            X_struct_trans = X_struct
            
        # Combine (sparse + dense = sparse via hstack)
        X_hybrid = hstack([X_text_vec, X_struct_trans])
        
        self.model.fit(X_hybrid, y_train)
        return X_hybrid
        
    def transform(self, X_text, X_struct):
        X_text_vec = self.vectorizer.transform(X_text)
        
        if self.use_scaler:
            X_struct_trans = self.scaler.transform(X_struct)
        else:
            X_struct_trans = X_struct
            
        return hstack([X_text_vec, X_struct_trans])
        
    def predict(self, X_text, X_struct):
        X_hybrid = self.transform(X_text, X_struct)
        return self.model.predict(X_hybrid)
        
    def save(self, filepath):
        joblib.dump({
            "vectorizer": self.vectorizer, 
            "scaler": self.scaler, 
            "model": self.model, 
            "use_scaler": self.use_scaler
        }, filepath)
        
    @classmethod
    def load(cls, filepath):
        data = joblib.load(filepath)
        instance = cls({}, {"model_type": "LogisticRegression"}) # dummy params
        instance.vectorizer = data["vectorizer"]
        instance.scaler = data.get("scaler")
        instance.use_scaler = data.get("use_scaler", False)
        instance.model = data["model"]
        return instance
