# Diabetes Classification App

**AUTHOR:** [Kevin Thomas](ket189@pitt.edu)

**CREATION DATE:** February 26, 2026 <br>
**UPDATE DATE:** February 26, 2026  

Data Source: [HERE](https://github.com/mytechnotalent/DRP-MLP)

Classify diabetes risk in real time with a PyTorch model and see predictions on an Arduino LED matrix by entering patient measurements via the web interface.

## Description

The App uses a Multi-Layer Perceptron (MLP) neural network trained on the DRP (Diabetes Risk Prediction) dataset to predict diabetes risk from sixteen input features: BMI, waist circumference, daily calorie intake, triglycerides level, fasting glucose level, HbA1c level, sugar intake, blood pressure, insulin level, cholesterol level, stress level, BMI category, physical activity level, glucose category, sleep hours, and age. Features are listed in order of importance as determined by their final Discriminative Score ($S$), which evaluates their ability to separate the classes by combining p-value strength ($P$), effect size ($E$), and normalized mutual information ($M$). Users can enter measurements through a web interface, and the prediction is visualized on the 8 x 13 LED matrix with unique patterns for each risk level.

The `assets` folder contains the **frontend** components of the application, including the HTML, CSS, and JavaScript files that make up the web user interface. The `python` folder contains the application **backend** with model inference and WebUI handling. The Arduino sketch manages LED matrix display.

## Bricks Used

The Diabetes Classification App uses the following Bricks:

- `arduino:web_ui`: Brick to create a web interface for inputting patient measurements and displaying predictions.

## Hardware and Software Requirements

### Hardware

- Arduino UNO Q (x1)
- USB-C® cable (for power and programming) (x1)

### Software

- Arduino App Lab
- PyTorch (for neural network inference)

## How to Use the Example

### Clone the Example

1. Clone the example to your workspace.

### Run the App

1. Click the **Run** button in App Lab to start the application.
2. Open the App in your browser at `<UNO-Q-IP-ADDRESS>:7000`
3. Enter the sixteen patient features (ranked by importance):
   - **BMI**: body mass index (e.g., `31.5`)
   - **Waist Circumference**: measurement in cm (e.g., `95.0`)
   - **Daily Calorie Intake**: daily calories (e.g., `2800.0`)
   - **Triglycerides Level**: triglycerides reading (e.g., `220.0`)
   - **Fasting Glucose Level**: fasting glucose reading (e.g., `120.0`)
   - **HbA1c Level**: glycated hemoglobin (e.g., `6.5`)
   - **Sugar Intake**: grams per day (e.g., `60.0`)
   - **Blood Pressure**: blood pressure reading (e.g., `140.0`)
   - **Insulin Level**: insulin reading (e.g., `15.0`)
   - **Cholesterol Level**: cholesterol reading (e.g., `240.0`)
   - **Stress Level**: scale 0-10 (e.g., `7.0`)
   - **BMI Category**: encoded category (e.g., `2.0`)
   - **Physical Activity Level**: encoded level (e.g., `1.0`)
   - **Glucose Category**: encoded category (e.g., `1.0`)
   - **Sleep Hours**: hours per night (e.g., `5.5`)
   - **Age**: patient age in years (e.g., `45.0`)
4. Click **Predict Risk** to see the result

### Input Validation

The web interface validates that all inputs are proper floats:
- Integers are rejected (e.g., `5` must be entered as `5.0`)
- Text/strings are rejected
- Valid format examples: `31.5`, `95.0`, `6.5`, `45.0`

### Example Measurements

Try these sample measurements to test each risk prediction:

| Risk Level | BMI  | Waist (cm) | Calories | Triglycerides | Fasting Glucose | HbA1c | Sugar (g/day) | Blood Pressure | Insulin | Cholesterol | Stress | BMI Cat | Activity | Glucose Cat | Sleep | Age  |
| ---------- | ---- | ---------- | -------- | ------------- | --------------- | ----- | ------------- | -------------- | ------- | ----------- | ------ | ------- | -------- | ----------- | ----- | ---- |
| High Risk  | 31.5 | 95.0       | 2800.0   | 220.0         | 120.0           | 6.5   | 60.0          | 140.0          | 15.0    | 240.0       | 7.0    | 2.0     | 1.0      | 1.0         | 5.5   | 45.0 |
| Low Risk   | 22.0 | 70.0       | 1800.0   | 100.0         | 85.0            | 5.0   | 20.0          | 120.0          | 8.0     | 180.0       | 3.0    | 1.0     | 2.0      | 0.0         | 7.5   | 30.0 |

## How it Works

Once the application is running, the device performs the following operations:

- **Serving the web interface and handling WebSocket communication.**

  The `web_ui` Brick provides the web server and WebSocket communication:

  ```python
  from arduino.app_bricks.web_ui import WebUI

  ui = WebUI()
  ui.on_message("predict", on_predict)
  ```

- **Loading the trained PyTorch model and scaler.**

  The application loads a pre-trained MLP model and StandardScaler for diabetes classification:

  ```python
  from arduino.app_utils import *
  import torch
  import torch.nn as nn
  import joblib

  model = Model().to(DEVICE)
  model.load_state_dict(torch.load("/app/python/drp_model.pth", map_location=DEVICE, weights_only=True))
  model.eval()

  scaler = joblib.load("/app/python/drp_scaler.pkl")
  ```

  The model and scaler are automatically loaded when the application starts and are ready to make predictions.

- **Making predictions based on input measurements.**

  The `predict_diabetes()` function takes sixteen features, scales them, and returns the predicted risk level:

  ```python
  def predict_diabetes(bmi: float, waist_circumference_cm: float,
                       daily_calorie_intake: float, triglycerides_level: float,
                       fasting_glucose_level: float, HbA1c_level: float,
                       sugar_intake_grams_per_day: float, blood_pressure: float,
                       insulin_level: float, cholesterol_level: float,
                       stress_level: float, bmi_category: float,
                       physical_activity_level: float, glucose_category: float,
                       sleep_hours: float, age: float) -> str:
      raw_features = _prepare_raw_features(
          bmi, waist_circumference_cm, daily_calorie_intake, triglycerides_level,
          fasting_glucose_level, HbA1c_level, sugar_intake_grams_per_day,
          blood_pressure, insulin_level, cholesterol_level, stress_level,
          bmi_category, physical_activity_level, glucose_category,
          sleep_hours, age)
      scaled_features = scaler.transform([raw_features])[0]
      X_new = torch.tensor(scaled_features).float().to(DEVICE)
      logits = model(X_new)
      predicted_class = logits.argmax(dim=1).item()
      return RISK_MAP[predicted_class]
  ```

  The model outputs one of two risk levels: `low_risk` or `high_risk`.

- **Handling web interface predictions and updating the LED matrix.**

  When a user submits measurements through the web interface:

  ```python
  def on_predict(client, data):
      risk = predict_diabetes(data["bmi"], data["waist_circumference_cm"],
                              data["daily_calorie_intake"], data["triglycerides_level"],
                              data["fasting_glucose_level"], data["HbA1c_level"],
                              data["sugar_intake_grams_per_day"], data["blood_pressure"],
                              data["insulin_level"], data["cholesterol_level"],
                              data["stress_level"], data["bmi_category"],
                              data["physical_activity_level"], data["glucose_category"],
                              data["sleep_hours"], data["age"])
      ui.send_message("prediction_result", {"risk": risk})
      Bridge.call("display_risk", risk)
  ```

- **Displaying risk patterns on the LED matrix.**

  The sketch receives the risk level and displays the corresponding pattern:

  ```cpp
  void display_risk(String risk) {
    if (risk == "low_risk") loadFrame8x13(low_risk);
    else if (risk == "high_risk") loadFrame8x13(high_risk);
    else loadFrame8x13(unknown);
  }
  ```

The high-level data flow looks like this:

```
Web Browser Input → WebSocket → Python Backend → PyTorch Model → Bridge → LED Matrix
```

- **`ui = WebUI()`**: Initializes the web server that serves the HTML interface and handles WebSocket communication.

- **`ui.on_message("predict", on_predict)`**: Registers a WebSocket message handler that responds when the user submits measurements.

- **`ui.send_message("prediction_result", ...)`**: Sends prediction results to the web client in real-time.

- **`RISK_MAP`**: Dictionary mapping class indices to risk level names (0 = low_risk, 1 = high_risk).

- **`DEVICE`**: Automatically selects the best available compute device (CUDA, MPS, or CPU).

- **`Model`**: A feedforward neural network class with two hidden layers (32 and 16 neurons) and optional dropout for regularization.

- **`predict_diabetes()`**: Takes sixteen float features (BMI, waist circumference, daily calorie intake, triglycerides level, fasting glucose level, HbA1c level, sugar intake, blood pressure, insulin level, cholesterol level, stress level, BMI category, physical activity level, glucose category, sleep hours, age), scales them using the fitted StandardScaler, runs inference through the model, and returns the predicted risk level name.

- **`Bridge.call("display_risk", risk)`**: Calls the Arduino function to update the LED matrix display.

### 🔧 Frontend (`index.html` + `app.js`)

The web interface provides a form for entering patient measurements with validation.

- **Socket.IO connection**: Establishes WebSocket communication with the Python backend through the `web_ui` Brick.

- **`socket.emit("predict", data)`**: Sends measurement data to the backend when the user clicks the predict button.

- **`socket.on("prediction_result", ...)`**: Receives prediction results and updates the UI accordingly.

- **`isValidFloat()`**: Validates that inputs are proper floats (rejects integers and strings).

### 🔧 Hardware (`sketch.ino`)

The Arduino code is focused on hardware management. It receives risk level names and displays them on the LED matrix.

- **`matrix.begin()`**: Initializes the matrix driver, making the LED display ready to show patterns.

- **`Bridge.begin()`**: Opens the serial communication bridge to the host Python® runtime.

- **`Bridge.provide("display_risk", display_risk)`**: Registers the display function to be callable from Python.

- **`display_risk(String risk)`**: Receives the predicted risk level and displays the corresponding 8 × 13 frame on the LED matrix.

- **`loop()`**: Waits for prediction updates from the Python backend.

- **`drp_frames.h`**: Header file that stores the pixel patterns for each diabetes risk level:
  - **Low Risk**: Checkmark pattern
  - **High Risk**: Exclamation mark pattern
  - **Unknown**: Question mark for error cases

## Neural Network Architecture

The MLP model consists of:

- **fc1**: Input (16) → Output (32), ReLU activation
- **fc2**: Input (32) → Output (16), ReLU activation
- **out**: Input (16) → Output (2), logits output

The model takes 16 input features (BMI, waist circumference, daily calorie intake, triglycerides level, fasting glucose level, HbA1c level, sugar intake, blood pressure, insulin level, cholesterol level, stress level, BMI category, physical activity level, glucose category, sleep hours, age) and outputs probabilities for 2 risk levels. Features are scaled using StandardScaler before inference.

## Author

**Kevin Thomas**

- Creation Date: February 27, 2026
- Last Updated: February 27, 2026
