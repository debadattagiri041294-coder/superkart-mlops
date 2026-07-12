import os
import joblib
import pandas as pd
import gradio as gr
from huggingface_hub import hf_hub_download

HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "your-username/superkart-sales-model")
MODEL_FILENAME = "model.joblib"

model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=MODEL_FILENAME, repo_type="model")
model = joblib.load(model_path)

def predict_sales(product_weight, product_sugar_content, product_allocated_area,
                   product_type, product_mrp, store_age, store_size,
                   store_location_city_type, store_type):
    input_df = pd.DataFrame([{
        "Product_Weight": product_weight,
        "Product_Sugar_Content": product_sugar_content,
        "Product_Allocated_Area": product_allocated_area,
        "Product_Type": product_type,
        "Product_MRP": product_mrp,
        "Store_Age": store_age,
        "Store_Size": store_size,
        "Store_Location_City_Type": store_location_city_type,
        "Store_Type": store_type,
    }])
    prediction = model.predict(input_df)[0]
    return f"Predicted Sales Total: {prediction:,.2f}"

demo = gr.Interface(
    fn=predict_sales,
    inputs=[
        gr.Number(label="Product Weight", value=12.5),
        gr.Dropdown(["Low Sugar", "Regular", "No Sugar"], label="Product Sugar Content", value="Low Sugar"),
        gr.Slider(0.0, 1.0, value=0.05, label="Product Allocated Area (ratio)"),
        gr.Dropdown(
            ["Meat", "Snack Foods", "Hard Drinks", "Dairy", "Canned", "Soft Drinks",
             "Health And Hygiene", "Baking Goods", "Bread", "Breakfast",
             "Frozen Foods", "Fruits And Vegetables", "Household", "Seafood",
             "Starchy Foods", "Others"],
            label="Product Type", value="Snack Foods",
        ),
        gr.Number(label="Product MRP", value=150.0),
        gr.Number(label="Store Age (years)", value=20),
        gr.Dropdown(["High", "Medium", "Small"], label="Store Size", value="Medium"),
        gr.Dropdown(["Tier 1", "Tier 2", "Tier 3"], label="Store Location City Type", value="Tier 2"),
        gr.Dropdown(
            ["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"],
            label="Store Type", value="Supermarket Type1",
        ),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="🛒 SuperKart Sales Forecast",
    description="Predict expected sales revenue for a product at a given store.",
)

demo.launch()
