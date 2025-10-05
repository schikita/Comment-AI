import streamlit as st
from ai.model import CommentModel
from ai.preprocess import clean_text, load_vectorizer

st.title("Определение тональности комментария")

vectorizer = load_vectorizer()
model = CommentModel.load()

comment = st.button("Проверить")

if st.button("Проверить"):
    text = clean_text(comment)
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    st.success("Комментарий +" if prediction == 1 else "Комеент -")
    


