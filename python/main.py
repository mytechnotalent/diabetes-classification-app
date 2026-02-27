"""
FILE: main.py

DESCRIPTION:
  Loads a trained PyTorch model for diabetes risk classification and provides prediction functionality.

BRIEF:
  Provides functions to predict diabetes risk based on patient health measurements.
  Allows a microcontroller to call these functions via Bridge to display predictions.

AUTHOR: Kevin Thomas
CREATION DATE: February 27, 2026
UPDATE DATE: February 27, 2026
"""

from arduino.app_utils import *
from arduino.app_bricks.web_ui import WebUI
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

# Risk level and device configuration
# Map class indices to diabetes risk level names
RISK_MAP = {0: "low_risk", 1: "high_risk"}

# Determine compute device: CUDA > MPS > CPU for optimal performance
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


class Model(nn.Module):
    """
    Feedforward neural network for diabetes risk classification.

    A two-layer feedforward network with optional dropout for regularization.
    Accepts 16 input features and outputs 2 class logits.

    ATTRIBUTES:
      fc1 (nn.Linear): First hidden layer (input -> h1).
      fc2 (nn.Linear): Second hidden layer (h1 -> h2).
      out (nn.Linear): Output layer (h2 -> num_classes).
      dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(self, in_features=16, h1=32, h2=16, out_features=2, dropout=0.0):
        """
        Initialize neural network layers.

        PARAMETERS:
          in_features (int): Number of input features (default: 16).
          h1 (int): Number of neurons in first hidden layer (default: 32).
          h2 (int): Number of neurons in second hidden layer (default: 16).
          out_features (int): Number of output classes (default: 2).
          dropout (float): Dropout rate, 0.0 = no dropout (default: 0.0).

        RETURNS:
          None
        """
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Forward pass through the network with ReLU activation and dropout.

        PARAMETERS:
          x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        RETURNS:
          torch.Tensor: Output logits of shape (batch_size, out_features).
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.out(x)


# Load pre-trained model and data scaler
# Initialize model, load trained weights, and set to evaluation mode
model = Model().to(DEVICE)
model.load_state_dict(
    torch.load("/app/python/drp_model.pth", map_location=DEVICE, weights_only=True)
)
model.eval()
# Load fitted scaler used during model training for feature normalization
scaler = joblib.load("/app/python/drp_scaler.pkl")


# Private helper functions for predict_diabetes (in call order)
def _prepare_raw_features(
    bmi: float,
    waist_circumference_cm: float,
    daily_calorie_intake: float,
    triglycerides_level: float,
    fasting_glucose_level: float,
    HbA1c_level: float,
    sugar_intake_grams_per_day: float,
    blood_pressure: float,
    insulin_level: float,
    cholesterol_level: float,
    stress_level: float,
    bmi_category: float,
    physical_activity_level: float,
    glucose_category: float,
    sleep_hours: float,
    age: float,
) -> list:
    """
    Prepare raw features in model input order.

    Accepts features in importance-ranked order and returns them reordered
    to match the scaler/model training order.

    PARAMETERS:
      bmi (float): Body mass index.
      waist_circumference_cm (float): Waist circumference in cm.
      daily_calorie_intake (float): Daily calorie intake.
      triglycerides_level (float): Triglycerides level.
      fasting_glucose_level (float): Fasting glucose level.
      HbA1c_level (float): Glycated hemoglobin level.
      sugar_intake_grams_per_day (float): Daily sugar intake in grams.
      blood_pressure (float): Blood pressure reading.
      insulin_level (float): Insulin level.
      cholesterol_level (float): Cholesterol level.
      stress_level (float): Stress level (0-10 scale).
      bmi_category (float): BMI category (encoded).
      physical_activity_level (float): Physical activity level (encoded).
      glucose_category (float): Glucose category (encoded).
      sleep_hours (float): Hours of sleep per night.
      age (float): Patient age in years.

    RETURNS:
      list: Features reordered to model input order [age, bmi,
            blood_pressure, fasting_glucose_level, insulin_level,
            HbA1c_level, cholesterol_level, triglycerides_level,
            physical_activity_level, daily_calorie_intake,
            sugar_intake_grams_per_day, sleep_hours, stress_level,
            waist_circumference_cm, bmi_category, glucose_category].
    """
    return [
        age,
        bmi,
        blood_pressure,
        fasting_glucose_level,
        insulin_level,
        HbA1c_level,
        cholesterol_level,
        triglycerides_level,
        physical_activity_level,
        daily_calorie_intake,
        sugar_intake_grams_per_day,
        sleep_hours,
        stress_level,
        waist_circumference_cm,
        bmi_category,
        glucose_category,
    ]


def _scale_and_tensorize(raw_features: list) -> torch.Tensor:
    """
    Scale features and convert to PyTorch tensor.

    PARAMETERS:
      raw_features (list): Raw feature values to scale.

    RETURNS:
      torch.Tensor: Scaled features as 2D tensor ready for model input.
    """
    scaled = scaler.transform([raw_features])[0]
    X = torch.tensor(scaled).float().to(DEVICE)
    return X.unsqueeze(0) if X.dim() == 1 else X


def _predict_class(X_tensor: torch.Tensor) -> int:
    """
    Get predicted class index from model logits.

    PARAMETERS:
      X_tensor (torch.Tensor): Scaled feature tensor.

    RETURNS:
      int: Predicted class index (0 or 1).
    """
    logits = model(X_tensor)
    return logits.argmax(dim=1).item()


def predict_diabetes(
    bmi: float,
    waist_circumference_cm: float,
    daily_calorie_intake: float,
    triglycerides_level: float,
    fasting_glucose_level: float,
    HbA1c_level: float,
    sugar_intake_grams_per_day: float,
    blood_pressure: float,
    insulin_level: float,
    cholesterol_level: float,
    stress_level: float,
    bmi_category: float,
    physical_activity_level: float,
    glucose_category: float,
    sleep_hours: float,
    age: float,
) -> str:
    """
    Predict diabetes risk from input features.

    Prepares features, scales them, and passes through the trained model to
    predict the diabetes risk level.

    PARAMETERS:
      bmi (float): Body mass index.
      waist_circumference_cm (float): Waist circumference in cm.
      daily_calorie_intake (float): Daily calorie intake.
      triglycerides_level (float): Triglycerides level.
      fasting_glucose_level (float): Fasting glucose level.
      HbA1c_level (float): Glycated hemoglobin level.
      sugar_intake_grams_per_day (float): Daily sugar intake in grams.
      blood_pressure (float): Blood pressure reading.
      insulin_level (float): Insulin level.
      cholesterol_level (float): Cholesterol level.
      stress_level (float): Stress level (0-10 scale).
      bmi_category (float): BMI category (encoded).
      physical_activity_level (float): Physical activity level (encoded).
      glucose_category (float): Glucose category (encoded).
      sleep_hours (float): Hours of sleep per night.
      age (float): Patient age in years.

    RETURNS:
      str: Predicted diabetes risk level name or error message.
    """
    try:
        raw_features = _prepare_raw_features(
            bmi,
            waist_circumference_cm,
            daily_calorie_intake,
            triglycerides_level,
            fasting_glucose_level,
            HbA1c_level,
            sugar_intake_grams_per_day,
            blood_pressure,
            insulin_level,
            cholesterol_level,
            stress_level,
            bmi_category,
            physical_activity_level,
            glucose_category,
            sleep_hours,
            age,
        )
        X_tensor = _scale_and_tensorize(raw_features)
        predicted_class = _predict_class(X_tensor)
        return RISK_MAP[predicted_class]
    except Exception as e:
        return f"Prediction Error: {str(e)}"


# Application interface setup
# Expose predict_diabetes function for direct microcontroller calls via Bridge protocol
Bridge.provide("predict_diabetes", predict_diabetes)

# Initialize web UI server on port 7000 for user interaction
ui = WebUI(port=7000)


# Private helper functions for on_predict (in call order)
def _extract_patient_measurements(data: dict) -> tuple:
    """
    Extract patient measurements from request data.

    PARAMETERS:
      data (dict): Request data containing patient measurements.

    RETURNS:
      tuple: (bmi, waist_circumference_cm, daily_calorie_intake,
              triglycerides_level, fasting_glucose_level, HbA1c_level,
              sugar_intake_grams_per_day, blood_pressure, insulin_level,
              cholesterol_level, stress_level, bmi_category,
              physical_activity_level, glucose_category, sleep_hours,
              age) with 0.0 defaults.
    """
    return (
        data.get("bmi", 0.0),
        data.get("waist_circumference_cm", 0.0),
        data.get("daily_calorie_intake", 0.0),
        data.get("triglycerides_level", 0.0),
        data.get("fasting_glucose_level", 0.0),
        data.get("HbA1c_level", 0.0),
        data.get("sugar_intake_grams_per_day", 0.0),
        data.get("blood_pressure", 0.0),
        data.get("insulin_level", 0.0),
        data.get("cholesterol_level", 0.0),
        data.get("stress_level", 0.0),
        data.get("bmi_category", 0.0),
        data.get("physical_activity_level", 0.0),
        data.get("glucose_category", 0.0),
        data.get("sleep_hours", 0.0),
        data.get("age", 0.0),
    )


def _send_result_to_clients(risk: str) -> None:
    """
    Send prediction result to web client and LED matrix display.

    PARAMETERS:
      risk (str): Predicted diabetes risk level name.

    RETURNS:
      None
    """
    ui.send_message("prediction_result", {"risk": risk})
    Bridge.call("display_risk", risk)


def on_predict(client, data):
    """
    Handle diabetes risk prediction request from web interface.

    Extracts measurements, runs prediction, and broadcasts result to web
    client and LED matrix display.

    PARAMETERS:
      client: The client connection requesting prediction.
      data (dict): Request data with bmi, waist_circumference_cm,
            daily_calorie_intake, triglycerides_level,
            fasting_glucose_level, HbA1c_level,
            sugar_intake_grams_per_day, blood_pressure, insulin_level,
            cholesterol_level, stress_level, bmi_category,
            physical_activity_level, glucose_category, sleep_hours, age.

    RETURNS:
      None
    """
    (
        bmi,
        waist_circumference_cm,
        daily_calorie_intake,
        triglycerides_level,
        fasting_glucose_level,
        HbA1c_level,
        sugar_intake_grams_per_day,
        blood_pressure,
        insulin_level,
        cholesterol_level,
        stress_level,
        bmi_category,
        physical_activity_level,
        glucose_category,
        sleep_hours,
        age,
    ) = _extract_patient_measurements(data)
    risk = predict_diabetes(
        bmi,
        waist_circumference_cm,
        daily_calorie_intake,
        triglycerides_level,
        fasting_glucose_level,
        HbA1c_level,
        sugar_intake_grams_per_day,
        blood_pressure,
        insulin_level,
        cholesterol_level,
        stress_level,
        bmi_category,
        physical_activity_level,
        glucose_category,
        sleep_hours,
        age,
    )
    _send_result_to_clients(risk)


# Script-level event handling and application initialization
# Register the on_predict handler to receive prediction requests via web socket
ui.on_message("predict", on_predict)

# Start the application main loop
App.run()
