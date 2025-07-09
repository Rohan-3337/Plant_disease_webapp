# 🌿 Plant Disease Classifier

A web-based application for plant disease detection using **DenseNet** and **Flask**, enhanced with **Groq AI** for real-time remedy suggestions rendered beautifully in markdown.

---

## 🚀 Features

* ✅ Upload leaf images and classify 38 plant diseases
* 🔬 Uses DenseNet with **Transfer Learning** for high accuracy
* 🤖 Remedies suggested using **Groq LLaMA3 AI**
* 📝 Markdown rendering of treatment steps on frontend
* 📊 99% accuracy on test dataset

---

## 📊 Model Evaluation

The model was trained on a plant disease dataset with 38 classes and evaluated on 10,849 test images.

| Metric    | Value         |
| --------- | ------------- |
| Accuracy  | 0.99          |
| Precision | 0.99          |
| Recall    | 0.99          |
| F1 Score  | 0.99          |
| Support   | 10,849 images |

---

## 🧠 Sample Classes (38 total)

* Apple\_\_\_Black\_rot
* Tomato\_\_\_Leaf\_Mold
* Corn\_(maize)\_\_\_Northern\_Leaf\_Blight
* Grape\_\_\_Esca\_(Black\_Measles)
* Potato\_\_\_Late\_blight
* ... and many more

---

## 🏧 Project Structure

```
plant-disease-classifier/
├── app.py
├── model/
│   └── densenet_model (1).keras
├── static/
│   └── uploads/
├── templates/
│   └── index.html
├── .env
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/plant-disease-classifier.git
cd plant-disease-classifier
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Groq API Key

Create a `.env` file in the root:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the Flask App

```bash
python app.py
```

Go to: `http://127.0.0.1:5000` in your browser

---

## 🖼️ How It Works

1. User uploads a leaf image
2. Model classifies the disease
3. Groq LLaMA3 API is called with the disease name
4. Remedy is rendered using markdown on the frontend

---

## 💬 Sample Remedy Output

**Prediction:** `Tomato___Leaf_Mold`

```markdown
- Remove infected leaves immediately.
- Improve air circulation around the plants.
- Avoid overhead watering.
- Apply copper-based fungicides if necessary.
```

---

## 📦 Requirements

```txt
Flask
Werkzeug
tensorflow
numpy
Pillow
groq
python-dotenv
```

---

## 🛡️ Security

* `.env` file is used to store the Groq API Key
* Ensure `.env` is listed in your `.gitignore`

```
.env
__pycache__/
*.pyc
```

---

## 📜 License

MIT © 2025 Rohan

---

## 🙌 Credits

* [Groq LLaMA3 API](https://console.groq.com/)
* [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)
* TensorFlow + Flask

---

## 📬 Contact

Made with 💚 by [Rohan](https://github.com/Rohan-3337)
