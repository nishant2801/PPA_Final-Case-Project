import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import joblib

def churn_rate(df, column, value):
    subset = df[df[column] == value]
    if len(subset) == 0:
        return 0
    return round(subset["churn"].mean() * 100, 2)

logistic_model = joblib.load("logistic_model.pkl")
rf_model = joblib.load("random_forest.pkl")
rf_features = joblib.load("feature_names.pkl")
logistic_features = ["restaurant_name_McDonald's", "dish_name_Fries"]
main_df = pd.read_csv("FooddeliveryApp Analysis Dataset.csv")
main_df["churn"] = main_df["churned"].map({"Active": 0, "Inactive": 1})
main_df["order_date"] = pd.to_datetime(main_df["order_date"], format="%m/%d/%Y", errors="coerce")
main_df["order_month"] = main_df["order_date"].dt.month
main_df["month_year"] = main_df["order_date"].dt.to_period("M").astype(str)


st.title("Customer Churn Prediction App")
st.write("Predict churn using Logistic Regression and Random Forest")
col1, col2 = st.columns(2)

with col1:
    restaurant = st.selectbox("Restaurant", ["KFC", "McDonald's", "Pizza Hut", "Subway", "Burger King"])
    dish = st.selectbox("Dish", ["Fries", "Pasta", "Pizza", "Sandwich", "Burger"])
    quantity = st.number_input("Quantity", min_value=1, max_value=200, value=2)
    price = st.number_input("Price", value=500.0)
    order_month = st.selectbox("Order Month", list(range(1, 13)))
    rating = st.number_input("Rating", min_value=1, max_value=5, value=3)

with col2:
    gender = st.selectbox("Gender", ["Male", "Other", "Female"])
    age = st.selectbox("Age Group", ["Senior", "Teenager", "Adult"])
    city = st.selectbox("City", ["Karachi", "Lahore", "Multan", "Peshawar", "Islamabad"])
    category = st.selectbox("Category", ["Continental", "Dessert", "Fast Food", "Italian" ,"Chinese"])
    payment = st.selectbox("Payment Method", ["Cash", "Wallet", "Card"])
    delivery = st.selectbox("Delivery Status", ["Delayed", "Delivered", "Cancelled"])

rf_input = pd.DataFrame(columns=rf_features)
rf_input.loc[0] = 0

rf_input.at[0, "quantity"] = quantity
rf_input.at[0, "price"] = price
rf_input.at[0, "rating"] = rating
rf_input.at[0, "order_month"] = order_month

def set_flag(df, col):
    if col in df.columns:
        df.at[0, col] = 1

set_flag(rf_input, f"gender_{gender}")
set_flag(rf_input, f"age_{age}")
set_flag(rf_input, f"city_{city}")
set_flag(rf_input, f"restaurant_name_{restaurant}")
set_flag(rf_input, f"dish_name_{dish}")
set_flag(rf_input, f"category_{category}")
set_flag(rf_input, f"payment_method_{payment}")
set_flag(rf_input, f"delivery_status_{delivery}")

log_input = pd.DataFrame(columns=logistic_features)
log_input.loc[0] = 0

set_flag(log_input, f"restaurant_name_{restaurant}")
set_flag(log_input, f"dish_name_{dish}")

st.subheader("📊 Churn Statistics Based on Selection")

colA, colB = st.columns(2)

with colA:
    rest_churn = churn_rate(main_df, "restaurant_name", restaurant)
    st.write(f"**{rest_churn}% customers churned from {restaurant}**")

    dish_churn = churn_rate(main_df, "dish_name", dish)
    st.write(f"**{dish_churn}% customers churned for dish: {dish}**")

    city_churn = churn_rate(main_df, "city", city)
    st.write(f"**{city_churn}% customers churned in {city}**")

with colB:
    cat_churn = churn_rate(main_df, "category", category)
    st.write(f"**{cat_churn}% customers churned in category: {category}**")

    del_churn = churn_rate(main_df, "delivery_status", delivery)
    st.write(f"**{del_churn}% orders with `{delivery}` status got churned**")

    age_churn = churn_rate(main_df, "age", age)
    st.write(f"**{age_churn}% customers in age group `{age}` churned**")


main_df["month_year"] = pd.to_datetime(main_df["month_year"])
monthly_churn = main_df.groupby("month_year")["churn"].sum()
monthly_churn = monthly_churn.round(2)
st.subheader("Monthly Churn Trend")

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(monthly_churn.index, monthly_churn.values, marker='o')
ax.set_xlabel("Month")
ax.set_ylabel("Churn Rate (%)")
ax.set_title("Customer Churn Trend Over Time")
ax.grid(True)
st.pyplot(fig)

if st.button("Predict"):
    log_pred = logistic_model.predict(log_input)[0]
    log_prob = logistic_model.predict_proba(log_input)[0][1]

    rf_pred = rf_model.predict(rf_input)[0]
    rf_prob = rf_model.predict_proba(rf_input)[0][1]

    st.subheader("Logistic Regression Prediction")
    st.write(f"Prediction: **{'Churn' if log_pred == 1 else 'Not Churn'}**")
    st.write(f"Probability of Churn: **{log_prob:.3f}**")

    st.write("---")
    st.subheader("Random Forest Prediction")
    st.write(f"Prediction: **{'Churn' if rf_pred == 1 else 'Not Churn'}**")
    st.write(f"Probability of Churn: **{rf_prob:.3f}**")
