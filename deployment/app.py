"""
app.py
------
Streamlit app for the SuperKart Sales Forecast model.
Deployed as a Hugging Face Space (Streamlit SDK).

Loads the trained pipeline (preprocessing + XGBoost model) from the
Hugging Face model hub, collects user inputs into a dataframe matching
the training schema, and returns a predicted sales total.
"""

import os
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "your-username/superkart-sales-model")
MODEL_FILENAME = "model.joblib"


@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=MODEL_FILENAME, repo_type="model")
    return joblib.load(model_path)


def main():
    st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="🛒")
    st.title("🛒 SuperKart Sales Forecast")
    st.write("Predict expected sales revenue for a product at a given store.")

    model = load_model()

    col1, col2 = st.columns(2)

    with col1:
        product_weight = st.number_input("Product Weight", min_value=0.0, value=12.5)
        product_sugar_content = st.selectbox(
            "Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"]
        )
        product_allocated_area = st.slider("Product Allocated Area (ratio)", 0.0, 1.0, 0.05)
        product_type = st.selectbox(
            "Product Type",
            [
                "Meat", "Snack Foods", "Hard Drinks", "Dairy", "Canned", "Soft Drinks",
                "Health And Hygiene", "Baking Goods", "Bread", "Breakfast",
                "Frozen Foods", "Fruits And Vegetables", "Household", "Seafood",
                "Starchy Foods", "Others",
            ],
        )
        product_mrp = st.number_input("Product MRP", min_value=0.0, value=150.0)
        store_age = st.number_input("Store Age (years)", min_value=0, value=20)

    with col2:
        store_size = st.selectbox("Store Size", ["High", "Medium", "Small"])
        store_location_city_type = st.selectbox(
            "Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"]
        )
        store_type = st.selectbox(
            "Store Type",
            ["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"],
        )

    if st.button("Predict Sales"):
        input_df = pd.DataFrame(
            [
                {
                    "Product_Weight": product_weight,
                    "Product_Sugar_Content": product_sugar_content,
                    "Product_Allocated_Area": product_allocated_area,
                    "Product_Type": product_type,
                    "Product_MRP": product_mrp,
                    "Store_Age": store_age,
                    "Store_Size": store_size,
                    "Store_Location_City_Type": store_location_city_type,
                    "Store_Type": store_type,
                }
            ]
        )

        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Sales Total: **{prediction:,.2f}**")


if __name__ == "__main__":
    main()
