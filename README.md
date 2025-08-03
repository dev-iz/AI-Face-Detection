# Real vs AI Face Classifier 🧠🤖

This project is a deep learning-based web app that detects whether a face image is **real** or **AI-generated** (e.g., from StyleGAN). Built using TensorFlow + Keras and deployed via Streamlit.

---

## 📌 Features

- Trains a binary image classifier (real vs fake faces)
- Uses transfer learning with ResNet50V2
- High precision and accuracy (99%+ on test set)
- Interactive prediction UI via Streamlit
- Upload your own image and get instant results

---

## 🛠️ Tech Stack

- **Framework**: TensorFlow + Keras
- **Model**: ResNet50V2 (with custom top layers)
- **Frontend**: Streamlit
- **Deployment**: Streamlit Cloud (or local)

---

## 🧪 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/your-username/real-or-ai-face-classifier.git
cd real-or-ai-face-classifier

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train the model (optional)
python3 real_or_ai_face_classifier_clean.py

# Launch Streamlit app
streamlit run streamlit_app.py
```

---

## 📁 Directory Structure

```
.
├── input/                        # Contains training data (ignored in Git)
│   └── AI-face-detection-Dataset/
├── model/                        # Contains trained model (auto-saved)
├── streamlit_app.py              # Streamlit frontend
├── real_or_ai_face_classifier_clean.py  # Training script
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git exclusions
└── README.md
```

---

## 🌍 Live Demo

> Coming soon: [https://your-username-your-repo.streamlit.app](https://your-username-your-repo.streamlit.app)

---

## 🤝 Contributions

Pull requests are welcome. For major changes, please open an issue first to discuss.

---

## 📜 License

[MIT](LICENSE)