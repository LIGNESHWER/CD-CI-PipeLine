"""Gradio application for the trained drug classification model."""

from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

MODEL_PATH = Path("model/drug_model.joblib")
model = joblib.load(MODEL_PATH)


def predict_drug(
    age: float,
    sex: str,
    blood_pressure: str,
    cholesterol: str,
    na_to_k: float,
) -> tuple[str, str]:
    """Return the predicted drug class and confidence."""
    if age < 0 or na_to_k < 0:
        return "Invalid input", "Age and Na_to_K must be non-negative."

    sample = pd.DataFrame(
        [
            {
                "Age": age,
                "Sex": sex,
                "BP": blood_pressure,
                "Cholesterol": cholesterol,
                "Na_to_K": na_to_k,
            }
        ]
    )
    prediction = str(model.predict(sample)[0])
    confidence = ""
    if hasattr(model, "predict_proba"):
        confidence = f"{max(model.predict_proba(sample)[0]) * 100:.2f}%"

    return prediction, confidence


demo = gr.Interface(
    fn=predict_drug,
    inputs=[
        gr.Number(label="Age", value=30, minimum=0),
        gr.Dropdown(["F", "M"], label="Sex", value="F"),
        gr.Dropdown(
            ["LOW", "NORMAL", "HIGH"],
            label="Blood Pressure Category",
            value="NORMAL",
        ),
        gr.Dropdown(
            ["NORMAL", "HIGH"],
            label="Cholesterol Category",
            value="NORMAL",
        ),
        gr.Number(label="Na_to_K Ratio", value=15.0, minimum=0),
    ],
    outputs=[
        gr.Textbox(label="Predicted Drug"),
        gr.Textbox(label="Model Confidence"),
    ],
    title="Drug Classification",
    description="Predict a drug class using the trained scikit-learn pipeline.",
)


if __name__ == "__main__":
    demo.launch()
