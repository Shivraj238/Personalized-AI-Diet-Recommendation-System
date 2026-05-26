import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os


st.set_page_config(page_title="NutriPlan AI", layout="wide", page_icon="🍎")


@st.cache_data
def load_data():
    if os.path.exists('food.csv'):
        df = pd.read_csv('food.csv')
        df.columns = df.columns.str.strip().str.capitalize()
        
        rename_map = {
            'Carbohydrates': 'Carbs', 
            'Dietary_type': 'Diet_Type', 
            'Category': 'Diet_Type', 
            'Type': 'Diet_Type',
            'Food_item': 'Food_Item'
        }
        df.rename(columns=rename_map, inplace=True)
        
        if 'Diet_Type' in df.columns:
            df['Diet_Type'] = df['Diet_Type'].astype(str).str.strip().str.title()
            df['Diet_Type'] = df['Diet_Type'].replace({
                'Vegetarian': 'Veg', 
                'Non-Vegetarian': 'Non-Veg', 
                'Non Veg': 'Non-Veg',
                'Veggie': 'Veg'
            })
        
        required_cols = ['Protein', 'Carbs', 'Calories', 'Fat', 'Diet_Type', 'Food_Item']
        for col in required_cols:
            if col not in df.columns:
                if col == 'Diet_Type':
                    df[col] = np.random.choice(['Veg', 'Non-Veg', 'Vegan'], len(df))
                elif col == 'Food_Item':
                    df[col] = [f"Recipe_{i}" for i in range(len(df))]
                else:
                    df[col] = np.random.randint(5, 100, len(df))
    else:
        np.random.seed(42)
        data = {
            'Food_Item': [f'Recipe_{i}' for i in range(1, 501)],
            'Calories': np.random.randint(50, 1000, 500),
            'Protein': np.random.randint(2, 50, 500),
            'Carbs': np.random.randint(0, 120, 500),
            'Fat': np.random.randint(1, 40, 500),
            'Diet_Type': np.random.choice(['Veg', 'Non-Veg', 'Vegan'], 500)
        }
        df = pd.DataFrame(data)
        
    X = df[['Protein', 'Carbs', 'Calories']]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)
    return df

df = load_data()


st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=100)
st.sidebar.title("Health Dashboard")
st.sidebar.markdown("---")
name = st.sidebar.text_input("👤 Name", "Guest")
weight = st.sidebar.number_input("⚖️ Weight (kg)", 30, 200, 70)
height = st.sidebar.number_input("📏 Height (cm)", 100, 250, 170)
goal = st.sidebar.selectbox("🎯 Fitness Goal", ["Weight Loss", "Muscle Gain", "Maintain Health"])


st.title("🍎 NutriPlan AI: Personalized Recommendations")
st.info(f"Welcome back, **{name}**! Configure your metrics in the sidebar to generate a new plan.")


tab1, tab2 = st.tabs(["🚀 Recommendation Engine", "📊 Data Insights"])

with tab1:
    st.header("Tailored Diet Plan")
    if st.button("✨ Generate My Personalized Plan", use_container_width=True):
        bmi = weight / ((height/100)**2)
        
       
        m1, m2, m3 = st.columns(3)
        m1.metric("Your BMI", f"{bmi:.2f}")
        
        if bmi < 18.5:
            m2.metric("Status", "Underweight", delta_color="inverse")
            st.warning("Your BMI suggests you are underweight. Consider increasing your caloric intake.")
        elif 18.5 <= bmi <= 24.9:
            m2.metric("Status", "Normal", delta_color="normal")
            st.success("Great! You have a healthy BMI.")
        else:
            m2.metric("Status", "Overweight", delta_color="inverse")
            st.warning("Your BMI suggests you are overweight. Focus on a balanced, calorie-controlled diet.")
            
        m3.metric("Goal", goal)
        
        # Algorithm Logic
        if goal == "Weight Loss":
            target_cluster = df.groupby('Cluster')['Calories'].mean().idxmin()
        elif goal == "Muscle Gain":
            target_cluster = df.groupby('Cluster')['Protein'].mean().idxmax()
        else:
            target_cluster = df.groupby('Cluster')['Calories'].mean().median()
            
        recommendations = df[df['Cluster'] == target_cluster].sample(min(3, len(df)))
        
        st.subheader("🥗 Your Daily Meal Suggestions")
        display_df = recommendations[['Food_Item', 'Calories', 'Protein', 'Carbs', 'Fat']].copy()
        display_df.columns = ["Recipe Name", "Calories", "Protein (g)", "Carbs (g)", "Fat (g)"]
        
        st.dataframe(display_df.style.highlight_max(axis=0, subset=["Protein (g)"], color='#d4edda'), 
                     use_container_width=True, hide_index=True)
        
        st.caption("Note: These suggestions are based on AI clustering of nutritional density.")

with tab2:
    st.header("Dataset Exploratory Analysis")
    st.markdown("Understanding how the AI groups food based on nutrient density.")
    
    sns.set(style="whitegrid")
    
    # Layout with columns
    c1, c2 = st.columns(2)
    
    with c1:
        st.write("**Nutrient Distribution**")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        sns.histplot(df['Calories'], bins=20, kde=True, color='#2ecc71', ax=ax1)
        st.pyplot(fig1)

        st.write("**Correlation Matrix**")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.heatmap(df[['Calories', 'Protein', 'Carbs', 'Fat']].corr(), annot=True, cmap='YlGnBu', ax=ax2)
        st.pyplot(fig2)

    with c2:
        st.write("**AI Clustering Visualization**")
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df, x='Protein', y='Calories', hue='Cluster', palette='viridis', ax=ax4)
        st.pyplot(fig4)
        
        st.write("**Dietary Variety**")
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        counts = df['Diet_Type'].value_counts()
        ax5.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['#2ecc71','#e74c3c','#3498db'])
        st.pyplot(fig5)