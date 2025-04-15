# app.py
import streamlit as st
from collections import defaultdict
import math

# =========================
# 🧠 Naive Bayes Classifier
# =========================
class NaiveBayes:
    def __init__(self):
        self.class_counts = defaultdict(int)
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.vocabulary = set()
        self.total_docs = 0

    def addExample(self, klass, words):
        self.class_counts[klass] += 1
        self.total_docs += 1
        for word in words:
            self.word_counts[klass][word] += 1
            self.vocabulary.add(word)

    def classify(self, words):
        log_probs = {}
        vocab_size = len(self.vocabulary)

        for klass in self.class_counts:
            log_probs[klass] = math.log(self.class_counts[klass] / self.total_docs)
            total_word_count = sum(self.word_counts[klass].values())

            for word in words:
                word_count = self.word_counts[klass][word]
                log_probs[klass] += math.log((word_count + 1) / (total_word_count + vocab_size))

        return max(log_probs, key=log_probs.get)

# =========================
# 🔁 App State Setup
# =========================
if "classes" not in st.session_state:
    st.session_state.classes = []
if "examples" not in st.session_state:
    st.session_state.examples = []
if "nb_model" not in st.session_state:
    st.session_state.nb_model = NaiveBayes()
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

# =========================
# 🧠 Streamlit Interface
# =========================
st.title("🧠 Interactive Naive Bayes Classifier Trainer")
st.markdown("Teach a Naive Bayes model using your own labels and examples!")

# =========================
# 1️⃣ Define Classes
# =========================
st.subheader("Step 1: Define Classification Labels")

with st.form("add_class_form", clear_on_submit=True):
    new_class = st.text_input("Add a new class (e.g. 'happy', 'sad', 'neutral')", key="new_class_input")
    submitted = st.form_submit_button("➕ Add Class")
    if submitted and new_class and new_class not in st.session_state.classes:
        st.session_state.classes.append(new_class)
        st.success(f"Added class '{new_class}'")

if st.session_state.classes:
    st.markdown(f"**Current Classes:** `{', '.join(st.session_state.classes)}`")

# =========================
# 2️⃣ Add Training Examples
# =========================
if st.session_state.classes:
    st.subheader("Step 2: Add Training Example")

    with st.form("add_example_form", clear_on_submit=True):
        example_sentence = st.text_input("Sentence")
        selected_class = st.selectbox("Label", st.session_state.classes)
        add_example = st.form_submit_button("🧠 Add Example")

        if add_example and example_sentence:
            st.session_state.examples.append((selected_class, example_sentence))
            st.success(f"Example added for class '{selected_class}'")

    if st.session_state.examples:
        st.markdown("### 📚 Training Examples")
        for idx, (klass, sentence) in enumerate(st.session_state.examples):
            st.write(f"{idx+1}. [{klass}] {sentence}")

# =========================
# 3️⃣ Train the Model
# =========================
if st.session_state.examples and not st.session_state.model_trained:
    st.subheader("Step 3: Train Model")
    if st.button("🚀 Train Naive Bayes"):
        st.session_state.nb_model = NaiveBayes()  # reset model
        for klass, sentence in st.session_state.examples:
            st.session_state.nb_model.addExample(klass, sentence.lower().split())
        st.session_state.model_trained = True
        st.success("✅ Model trained successfully!")

# =========================
# 4️⃣ Predict a Class
# =========================
if st.session_state.model_trained:
    st.subheader("Step 4: Test the Model")
    test_sentence = st.text_input("🧪 Enter a sentence to classify:")
    if st.button("🔍 Predict Class"):
        if test_sentence:
            pred = st.session_state.nb_model.classify(test_sentence.lower().split())
            st.success(f"📊 The predicted class is: **{pred}**")
        else:
            st.warning("⚠️ Enter a sentence first.")

# =========================
# 5️⃣ Reset Everything
# =========================
st.divider()
if st.button("🔄 Reset Everything"):
    st.session_state.clear()
    st.rerun()