

<h1 align="center">🛡️ AI-Based Intrusion Detection System</h1>

<p align="center">
  <b>Machine Learning Powered Network Security System</b><br>
  Detect • Classify • Prevent Cyber Attacks using AI
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python">
  <img src="https://img.shields.io/badge/DL-Tensorflow keras-orange?logo=Tensorflow">
  <img src="https://img.shields.io/badge/Framework-Streamlit-black?logo=Streamlit">
  <img src="https://img.shields.io/badge/Status-Active-success">
  <img src="https://img.shields.io/github/stars/gurupraneshNC/AI-based-IDS?logo=github">
</p>

---

# 📌 Project Overview

The **AI-Based Intrusion Detection System (IDS)** is a Machine Learning driven security system designed to detect malicious network activity and classify different types of cyber attacks.

Traditional IDS systems rely on static rules and signatures.  
This project enhances detection capability using **supervised learning models** to improve accuracy and adaptability.

---



# 🧠 Features

✅ Binary Classification (Normal vs Attack)  
✅ Multi-Class Attack Detection  
✅ Data Preprocessing & Feature Engineering  
✅ Model Evaluation Metrics  
✅ Confusion Matrix Visualization  
✅ Web Application Interface  
✅ Modular Notebook Architecture  
✅ Saved Trained Models  

---

# 🏗️ System Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/gurupraneshNC/AI-based-IDS/refs/heads/main/images/sys_arch.png" width="750">
</p>

### Flow:

```
Network Data → Preprocessing → Feature Engineering → ML Model → Prediction → Alert
```

---

# 📂 Project Structure

```
AI-based-IDS/
│
├── Notebooks/
│   ├── BinaryClassifiers.ipynb
│   ├── MultiClassClassifiers.ipynb
│
├── WebAPP/
│   ├── app.py
│   ├── templates/
│   └── static/
│
├── models/
│   └── cnn_binary.keras
|   └── cnnmulti.keras
|   └── lstm_binary.keras
|   └── lstm_multi.keras
│
├── requirements.txt
└── README.md
```

---

# 🛠️ Technologies Used

| Category | Tools |
|----------|--------|
| Programming | Python |
| Data Processing | NumPy, Pandas |
| Deep Learning | TensorFlow, Keras |
| Visualization | Matplotlib, Seaborn |
| Web Framework | Streamlit |
| Notebook | Jupyter |

---

# 📊 Deep Learning Models Used

- Long Short Term Memory (LSTM)
- Convolutional Neural Network (CNN)

---

# 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

# 🚀 Installation Guide

## 1️⃣ Clone Repository

```bash
git clone https://github.com/gurupraneshNC/AI-based-IDS.git
cd AI-based-IDS
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### Activate:

Windows:
```bash
venv\Scripts\activate
```

Linux/Mac:
```bash
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🧪 Run Jupyter Notebooks(Otional)

```bash
jupyter notebook
```

Open:
```
Notebooks/BinaryClassifiers.ipynb
```

or

```
Notebooks/MultiClassClassifiers.ipynb
```

---

# 🌐 Run Web Application

```bash
cd WebAPP
Streamlit run app.py
```

Open browser:

```
http://127.0.0.1:5000
```

---

# 📚 Dataset Used

The system is trained on benchmark intrusion detection datasets such as:

- UNSW-NB15

Dataset includes:
- Normal traffic
- DoS attacks
- Probe attacks
- R2L
- U2R
- etc.,

---

# 🔍 How It Works

1. Raw network data is loaded
2. Data cleaning & preprocessing applied
3. Feature scaling & encoding performed
4. DL model trained
5. Model saved
6. Web interface loads model
7. User inputs traffic features
8. System predicts attack type

---

# 🛡️ Why AI-Based IDS?

Traditional IDS:
- Signature based
- Limited adaptability
- Requires manual updates

AI-Based IDS:
- Learns from data
- Detects unseen patterns
- Adaptive & scalable
- Higher detection accuracy

---

# 📊 Future Improvements

- Real-time packet sniffing integration
- Integration with SIEM tools
- Cloud deployment (AWS/Azure)
- API endpoints for enterprise use

---

# 🤝 Contribution Guidelines

Contributions are welcome!

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push branch
5. Open Pull Request

---

# 📬 Contact

👤 **Gurupranesh Kulkarni**  
🔗 GitHub: https://github.com/gurupraneshNC  

For collaborations or queries, open an issue.

---

# 📜 License

This project is licensed under the MIT License.

---

# ⭐ Support

If you found this project useful:

⭐ Star this repository  
🔁 Share with security enthusiasts  
💡 Contribute to improve detection systems  

---

<p align="center">
  <b>🔐 Securing Networks with Intelligence 🔐</b>
</p>

<p align="center">
  Made with ❤️ by Gurupranesh
</p>

