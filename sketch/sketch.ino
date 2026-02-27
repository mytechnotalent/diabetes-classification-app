/**
 * FILE: sketch.ino
 *
 * DESCRIPTION:
 *   Displays diabetes risk prediction on an 8x13 LED matrix using Arduino.
 *
 * BRIEF:
 *   Loads frames representing diabetes risk levels on the matrix.
 *   Uses Bridge to fetch prediction from Python backend.
 *
 * AUTHOR: Kevin Thomas
 * CREATION DATE: February 27, 2026
 * UPDATE DATE: February 27, 2026
 */

#include <Arduino_RouterBridge.h>
#include "drp_frames.h"
#include "led_matrix.h"

/**
 * Private helper function to load risk frame by risk level name.
 *
 * PARAMETERS:
 *   risk (String): The predicted risk level name.
 *
 * RETURN:
 *   void
 */
void _load_risk_frame(String risk)
{
    if (risk == "low_risk")
        loadFrame8x13(low_risk);
    else if (risk == "high_risk")
        loadFrame8x13(high_risk);
    else
        loadFrame8x13(unknown);
}

/**
 * Initialize LED matrix and Bridge communication.
 *
 * RETURN:
 *   void
 */
void setup()
{
    matrix.begin();
    matrix.clear();
    Bridge.begin();
    Bridge.provide("display_risk", display_risk);
}

/**
 * Main loop waiting for prediction updates from Python backend.
 *
 * RETURN:
 *   void
 */
void loop()
{
    delay(100);
}

/**
 * Display predicted diabetes risk level on LED matrix.
 *
 * PARAMETERS:
 *   risk (String): The predicted diabetes risk level name.
 *
 * RETURN:
 *   void
 */
void display_risk(String risk)
{
    _load_risk_frame(risk);
}
