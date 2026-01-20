import gradio as gr
import pandas as pd
import pickle


with open("best_model_loan.pkl", "rb") as f:
    model = pickle.load(f)

def predict_loan(
    Gender, Married, Dependents, Education, Self_Employed,
    ApplicantIncome, CoapplicantIncome, LoanAmount,
    Loan_Amount_Term, Credit_History, Property_Area
):
    input_df = pd.DataFrame([[
        Gender, Married, Dependents, Education, Self_Employed,
        ApplicantIncome, CoapplicantIncome, LoanAmount,
        Loan_Amount_Term, Credit_History, Property_Area
    ]], columns=[
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
        'Loan_Amount_Term', 'Credit_History', 'Property_Area'
    ])

    pred = model.predict(input_df)[0]
    return "Loan Approved" if pred == 1 else "Loan Rejected"

inputs = [
    gr.Radio(["Male", "Female"], label="Gender"),
    gr.Radio(["Yes", "No"], label="Married"),
    gr.Dropdown(["0", "1", "2", "3+"], label="Dependents"),
    gr.Radio(["Graduate", "Not Graduate"], label="Education"),
    gr.Radio(["Yes", "No"], label="Self Employed"),

    gr.Slider(0, 100000, step=500, label="Applicant Income"),
    gr.Slider(0, 50000, step=500, label="Coapplicant Income"),

    gr.Slider(0, 700, step=10, label="Loan Amount"),
    gr.Slider(12, 480, step=12, label="Loan Amount Term"),
    gr.Radio([0, 1], label="Credit History"),
    gr.Dropdown(["Urban", "Semiurban", "Rural"], label="Property Area")
]

app = gr.Interface(
    fn=predict_loan,
    inputs=inputs,
    outputs="text",
    title="Loan Prediction System"
)

app.launch(share=True)

