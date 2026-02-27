/**
 * FILE: app.js
 *
 * DESCRIPTION:
 *   Web UI for diabetes classification app.
 *   Handles form submission, validation, socket communication, and result display.
 *
 * BRIEF:
 *   Manages client-side interactions for diabetes prediction input and output.
 *   Uses WebSocket (Socket.IO) to communicate with Arduino backend.
 *
 * AUTHOR: Kevin Thomas
 * CREATION DATE: February 27, 2026
 * UPDATE DATE: February 27, 2026
 *
 * SPDX-FileCopyrightText: Copyright (C) Kevin Thomas
 * SPDX-License-Identifier: MPL-2.0
 */

// Initialize Socket.IO connection to server
const socket = io(`http://${window.location.host}`);

// Reference to error container element for connection status messages
let errorContainer;

/**
 * Initialize event listeners when DOM is fully loaded.
 */
document.addEventListener('DOMContentLoaded', () => {
    errorContainer = document.getElementById('error-container');
    _init_socket_events();
    _init_form_handler();
});

/**
 * Private helper to handle connection established event.
 */
function _handle_connect() {
    if (errorContainer) {
        errorContainer.style.display = 'none';
        errorContainer.textContent = '';
    }
}

/**
 * Private helper to handle prediction result received.
 *
 * PARAMETERS:
 *   message (object): Message object containing risk prediction.
 */
function _handle_prediction_result(message) {
    _display_result(message.risk);
}

/**
 * Private helper to handle connection lost event.
 */
function _handle_disconnect() {
    if (errorContainer) {
        errorContainer.textContent = 'Connection to the board lost. Please check the connection.';
        errorContainer.style.display = 'block';
    }
}

/**
 * Initialize Socket.IO event listeners for connection and messages.
 */
function _init_socket_events() {
    socket.on('connect', _handle_connect);
    socket.on('prediction_result', _handle_prediction_result);
    socket.on('disconnect', _handle_disconnect);
}

/**
 * Initialize form submission event handler.
 */
function _init_form_handler() {
    const form = document.getElementById('diabetes-form');
    form.addEventListener('submit', _on_form_submit);
}

/**
 * Validate if value is a valid floating-point number.
 *
 * Must have decimal point; rejects pure integers and empty strings.
 *
 * PARAMETERS:
 *   value (string): The value to validate.
 *
 * RETURN:
 *   boolean: true if valid float, false otherwise.
 */
function _is_valid_float(value) {
    const trimmed = value.trim();
    if (trimmed === '') return false;
    if (/^-?\d+$/.test(trimmed)) return false;
    return /^-?\d*\.\d+$/.test(trimmed);
}

/**
 * Private helper to validate all form input fields.
 *
 * Updates error display states for invalid fields.
 *
 * PARAMETERS:
 *   fields (array): Array of field objects with id and errorId.
 *
 * RETURN:
 *   boolean: true if all fields valid, false otherwise.
 */
function _validate_form_fields(fields) {
    let valid = true;
    fields.forEach(field => {
        const input = document.getElementById(field.id);
        const error = document.getElementById(field.errorId);
        if (!_is_valid_float(input.value)) {
            input.classList.add('error');
            error.style.display = 'block';
            valid = false;
        } else {
            input.classList.remove('error');
            error.style.display = 'none';
        }
    });
    return valid;
}

/**
 * Private helper to collect and parse form input values.
 *
 * RETURN:
 *   object: Object with bmi, waist_circumference_cm, daily_calorie_intake,
 *           triglycerides_level, fasting_glucose_level, HbA1c_level,
 *           sugar_intake_grams_per_day, blood_pressure, insulin_level,
 *           cholesterol_level, stress_level, bmi_category,
 *           physical_activity_level, glucose_category, sleep_hours,
 *           age properties.
 */
function _collect_form_data() {
    return {
        bmi: parseFloat(document.getElementById('bmi').value),
        waist_circumference_cm: parseFloat(document.getElementById('waist-circumference').value),
        daily_calorie_intake: parseFloat(document.getElementById('daily-calorie-intake').value),
        triglycerides_level: parseFloat(document.getElementById('triglycerides-level').value),
        fasting_glucose_level: parseFloat(document.getElementById('fasting-glucose-level').value),
        HbA1c_level: parseFloat(document.getElementById('hba1c-level').value),
        sugar_intake_grams_per_day: parseFloat(document.getElementById('sugar-intake').value),
        blood_pressure: parseFloat(document.getElementById('blood-pressure').value),
        insulin_level: parseFloat(document.getElementById('insulin-level').value),
        cholesterol_level: parseFloat(document.getElementById('cholesterol-level').value),
        stress_level: parseFloat(document.getElementById('stress-level').value),
        bmi_category: parseFloat(document.getElementById('bmi-category').value),
        physical_activity_level: parseFloat(document.getElementById('physical-activity-level').value),
        glucose_category: parseFloat(document.getElementById('glucose-category').value),
        sleep_hours: parseFloat(document.getElementById('sleep-hours').value),
        age: parseFloat(document.getElementById('age').value)
    };
}

/**
 * Handle form submission and validate before sending prediction request.
 *
 * PARAMETERS:
 *   e (event): The form submission event.
 */
function _on_form_submit(e) {
    e.preventDefault();
    const fields = [
        { id: 'bmi', errorId: 'bmi-error' },
        { id: 'waist-circumference', errorId: 'wc-error' },
        { id: 'daily-calorie-intake', errorId: 'dci-error' },
        { id: 'triglycerides-level', errorId: 'tl-error' },
        { id: 'fasting-glucose-level', errorId: 'fgl-error' },
        { id: 'hba1c-level', errorId: 'hba1c-error' },
        { id: 'sugar-intake', errorId: 'si-error' },
        { id: 'blood-pressure', errorId: 'bp-error' },
        { id: 'insulin-level', errorId: 'il-error' },
        { id: 'cholesterol-level', errorId: 'cl-error' },
        { id: 'stress-level', errorId: 'stl-error' },
        { id: 'bmi-category', errorId: 'bc-error' },
        { id: 'physical-activity-level', errorId: 'pal-error' },
        { id: 'glucose-category', errorId: 'gc-error' },
        { id: 'sleep-hours', errorId: 'sh-error' },
        { id: 'age', errorId: 'age-error' }
    ];
    if (_validate_form_fields(fields)) {
        socket.emit('predict', _collect_form_data());
        document.getElementById('result').style.display = 'none';
    }
}

/**
 * Display prediction result to user on the web UI.
 *
 * PARAMETERS:
 *   risk (string): The predicted diabetes risk level name.
 */
function _display_result(risk) {
    const resultDiv = document.getElementById('result');
    const resultText = document.getElementById('result-text');
    resultText.textContent = `Predicted Risk: ${risk}`;
    resultDiv.style.display = 'block';
}
