import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from model import CommentModel
from preprocess import clean_text, train_vectorizer

def train_model(csv_path: str = "data/comments.csv"):
    print("загрузка данных...")
    df = pd.read_csv(csv_path)
    
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV должен содержать колонки: 'text' и 'label'")
    
    df["clean_text"] = df["text"].apply(clean_text)
    
    print("Обучаем векторизатор...")
    X = train_vectorizer(df["clean_text"])
    y = df["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state=42)
    
    model = CommentModel()
    model.train(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Точность: {acc:.2%}")
    print(classification_report(y_test, y_pred))
    
    model.save()
    print("Модель сохранена в 'ai/comment_model.pkl'")
    print("Векторизатор сохранен в 'ai/vectorizer.pkl'")
    
if __name__ == "__main__":
    train_model()
    
    