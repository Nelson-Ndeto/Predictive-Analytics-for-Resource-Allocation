

# Predictive Analytics for Resource Allocation  
## Random Forest Classifier + Confusion Matrix + Feature Importance Plot  
## 📌 Overview  
This project demonstrates how machine learning can support predictive analytics in software engineering, specifically for resource allocation. Using a **Random Forest Classifier**, the model predicts whether a case is *benign* or *malignant* based on structured numerical features.  
The project also visualizes:  
- A **confusion matrix**  
- **Feature importance rankings**  

---

## 📂 Files Included  
- `random_forest_with_graphs.py` — Main script  
- `breast_cancer_data.csv` — Dataset used for training and evaluation  

---

## ⚙️ Requirements

Install dependencies:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

---

## 🚀 How to Run (VS Code)

1. Place `breast_cancer_data.csv` in the same folder as the script.  
2. Run the Python file:

```bash
python random_forest_with_graphs.py
```

3. The script will:
   - Load and preprocess the dataset  
   - Train a Random Forest model  
   - Evaluate accuracy & F1-score  
   - Display:
     - Confusion Matrix  
     - Feature Importance Plot  

---

## 🧠 What the Script Does

### **1. Load Dataset**  
Reads `breast_cancer_data.csv` and prints a preview.

### **2. Label Encoding**  
Converts diagnosis labels:  
- `benign` → **0**  
- `malignant` → **1**

### **3. Train-Test Split**  
Splits data into 80% training and 20% testing.

### **4. Model Training**  
Uses a **Random Forest Classifier (100 trees)**.

### **5. Evaluation Metrics**  
Outputs:  
- **Accuracy**  
- **F1-score**

### **6. Visualizations**  
✔ Confusion matrix (Blue heatmap)  
✔ Feature importance (Horizontal bar chart)

---

## 📊 Example Outputs

### Sample Evaluation:
```
Accuracy: 0.9533
F1 Score: 0.9481
```

### Visuals Generated:
- Confusion Matrix  
- Feature Importance Plot  

---

## 📝 Notes
- Ensure the dataset is inside the project folder.  
- You can replace the dataset with any similar structured data for testing.  

---