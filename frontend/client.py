import streamlit as st
import requests
import time

# Local
# WEB_SERVER_URL = 'http://localhost:8000'
# Docker
WEB_SERVER_URL = "http://webserver:8000"

st.set_page_config(page_title="NexDish 🍱", layout="centered")
st.title("🥗 NexDish - Smart Food Classifier")
st.markdown("Upload a food image to get insights about the dish, its origin, and nutritional values.")

uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

# Clear output when a new file is uploaded
if uploaded_file:
    st.image(uploaded_file, caption="📸 Uploaded Image", width=300)
    
    # Create placeholder for status
    status_placeholder = st.empty()
    
    # Show "classifying..." message first
    status_placeholder.info("🔍 Classifying image...")
    time.sleep(2)
    status_placeholder.info("🔄 Generating response...")
    
    try:
        # Send image to backend (API call starts immediately)
        response = requests.post(
            f"{WEB_SERVER_URL}/upload",
            files={"file": uploaded_file.getvalue()}
        )
        
        # Wait 3 seconds and change message (API call is already running)
   
        
        if response.status_code == 200:
            data = response.json()
            status_placeholder.empty()
            
            # Create placeholder for prediction
            prediction_placeholder = st.empty()
            
            # Typing animation for predicted class
            predicted_class = data.get('predicted_class', 'Unknown').capitalize()
            for i in range(1, len(predicted_class) + 1):
                prediction_placeholder.success(f"🍽️ **Predicted Dish:** `{predicted_class[:i]}`")
                time.sleep(0.05)
            
            # Get food info data
            food_info = data.get("food_info", {})
            
            # Add the description header FIRST
            st.markdown("### 📝 Description")
            
            # Create placeholder for description and animate
            description_placeholder = st.empty()
            description = food_info.get("description", "N/A")
            for i in range(1, len(description) + 1):
                description_placeholder.markdown(description[:i])
                time.sleep(0.002)
            
            # Nutritional Highlights with typing animation
            st.markdown("### 💪 Nutritional Highlights")
            
            # Animate each nutrition field
            calories_placeholder = st.empty()
            calories_text = f"**Calories:** {food_info.get('calories', 'N/A')}"
            for i in range(1, len(calories_text) + 1):
                calories_placeholder.markdown(calories_text[:i])
                time.sleep(0.002)
            
            macros_placeholder = st.empty()
            macros_text = f"**Macronutrients:** {food_info.get('macronutrients', 'N/A')}"
            for i in range(1, len(macros_text) + 1):
                macros_placeholder.markdown(macros_text[:i])
                time.sleep(0.002)
                
            micros_placeholder = st.empty()
            micros_text = f"**Micronutrients:** {food_info.get('micronutrients', 'N/A')}"
            for i in range(1, len(micros_text) + 1):
                micros_placeholder.markdown(micros_text[:i])
                time.sleep(0.002)
                
            insights_placeholder = st.empty()
            insights_text = f"**Nutritional Insights:** {food_info.get('nutritional_insights', 'N/A')}"
            for i in range(1, len(insights_text) + 1):
                insights_placeholder.markdown(insights_text[:i])
                time.sleep(0.002)
                
            risks_placeholder = st.empty()
            risks_text = f"**Health Risks:** ⚠️ {food_info.get('health_risks', 'N/A')}"
            for i in range(1, len(risks_text) + 1):
                risks_placeholder.markdown(risks_text[:i])
                time.sleep(0.002)
            
            # Cultural Insights with typing animation
            st.markdown("### 🌍 Cultural Insights")
            culture_placeholder = st.empty()
            culture_text = food_info.get("cultural_insights", "N/A")
            for i in range(1, len(culture_text) + 1):
                culture_placeholder.markdown(culture_text[:i])
                time.sleep(0.002)
            
            # Food Profile with typing animation
            st.markdown("### 🍽️ Food Profile")
            col1, col2 = st.columns(2)
            
            with col1:
                cat_placeholder = st.empty()
                cat_text = f"**Category:** {food_info.get('category', 'N/A')}"
                for i in range(1, len(cat_text) + 1):
                    cat_placeholder.markdown(cat_text[:i])
                    time.sleep(0.002)
                    
                diet_placeholder = st.empty()
                diet_text = f"**Diet Suitability:** {food_info.get('diet_suitability', 'N/A')}"
                for i in range(1, len(diet_text) + 1):
                    diet_placeholder.markdown(diet_text[:i])
                    time.sleep(0.002)
                    
                season_placeholder = st.empty()
                season_text = f"**Seasonality:** {food_info.get('seasonality', 'N/A')}"
                for i in range(1, len(season_text) + 1):
                    season_placeholder.markdown(season_text[:i])
                    time.sleep(0.002)
                    
            with col2:
                allergen_placeholder = st.empty()
                allergen_text = f"**Common Allergens:** {food_info.get('common_allergens', 'N/A')}"
                for i in range(1, len(allergen_text) + 1):
                    allergen_placeholder.markdown(allergen_text[:i])
                    time.sleep(0.002)
                    
                fact_placeholder = st.empty()
                fact_text = f"**Interesting Fact:** {food_info.get('interesting_facts', 'N/A')}"
                for i in range(1, len(fact_text) + 1):
                    fact_placeholder.markdown(fact_text[:i])
                    time.sleep(0.002)

        elif response.status_code == 400:
            data = response.json()
            if "warning" in data and "outlier_score" in data:
                st.warning(f"⚠️ Outlier Warning - Please upload a valid food photo: {data['warning']} (score={data['outlier_score']:.2f})")
            else:
                st.error(f"❌ Bad request: {response.text}")
        else:
                st.error(f"❌ Server Error: {response.text}")

    except Exception as e:
        status_placeholder.error(f"❌ Failed to connect to API: {str(e)}")