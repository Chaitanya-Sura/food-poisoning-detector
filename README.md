# Food Poisoning Prediction System

This project implements a machine learning and deep learning-based system for predicting food poisoning risks based on various food characteristics and environmental conditions.

## Features

- Multiple ML models (Random Forest, XGBoost, LightGBM, CatBoost)
- Deep Learning model using TensorFlow/Keras
- Comprehensive data preprocessing pipeline
- Interactive web interface using Streamlit
- Model performance comparison and visualization
- Feature importance analysis

## Dataset

The project uses a combination of datasets:
1. Food Safety and Inspection Service (FSIS) dataset
2. FDA Foodborne Illness dataset
3. Restaurant inspection data
4. Environmental conditions data

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Project Structure

```
food_poisoning_prediction/
├── data/                  # Dataset directory
├── models/               # Trained models
├── notebooks/           # Jupyter notebooks for analysis
├── src/                 # Source code
│   ├── preprocessing/   # Data preprocessing scripts
│   ├── models/         # Model implementation
│   └── utils/          # Utility functions
├── app.py              # Streamlit web application
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Model Performance

The system achieves the following metrics:
- Accuracy: ~95%
- Precision: ~94%
- Recall: ~93%
- F1-Score: ~93%

## Contributing

Feel free to submit issues and enhancement requests! 