from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib

class StructureBaseline:
    def __init__(self, model_params, use_scaler=False):
        self.use_scaler = use_scaler
        self.scaler = StandardScaler() if use_scaler else None
        
        self.model_params = model_params.copy()
        m_type = self.model_params.pop("model_type")
        
        if m_type == "LogisticRegression":
            self.model = LogisticRegression(**self.model_params)
        elif m_type == "RandomForestClassifier":
            self.model = RandomForestClassifier(**self.model_params)
        elif m_type == "HistGradientBoostingClassifier":
            self.model = HistGradientBoostingClassifier(**self.model_params)
        else:
            raise ValueError(f"Unknown model type: {m_type}")
            
    def fit(self, X_train, y_train):
        if self.use_scaler:
            X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)
        return X_train
        
    def transform(self, X):
        if self.use_scaler:
            return self.scaler.transform(X)
        return X
        
    def predict(self, X):
        X_trans = self.transform(X)
        return self.model.predict(X_trans)
        
    def predict_proba(self, X):
        X_trans = self.transform(X)
        if not hasattr(self.model, "predict_proba"):
            raise NotImplementedError("This model does not support probability predictions.")
        return self.model.predict_proba(X_trans)
        
    def save(self, filepath):
        joblib.dump({"scaler": self.scaler, "model": self.model, "use_scaler": self.use_scaler}, filepath)
        
    @classmethod
    def load(cls, filepath):
        data = joblib.load(filepath)
        instance = cls({"model_type": "LogisticRegression"}) # dummy params
        instance.scaler = data.get("scaler")
        instance.use_scaler = data.get("use_scaler", False)
        instance.model = data["model"]
        return instance
