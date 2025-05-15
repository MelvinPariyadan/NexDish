import streamlit as st
import requests

#local
#WEB_SERVER_URL = 'http://localhost:8000'

#docker
WEB_SERVER_URL = "http://webserver:8000"


st.set_page_config(page_title="NexDish 🍱", layout="centered")

st.title("🥗 NexDish - Smart Food Classifier")
st.markdown("Upload a food image to get insights about the dish, its origins, and nutritional values.")

uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="📸 Uploaded Image", width=300)
    st.write("🔍 Classifying...")

    try:
        # Send image to backend
        response = requests.post(
            f"{WEB_SERVER_URL}/upload",
            files={"file": uploaded_file.getvalue()}
        )

        if response.status_code == 200:
            data = response.json()

            st.success(f"🍽️ **Predicted Dish:** `{data['predicted_class'].capitalize()}`")

            food_info = data.get("food_info", {})

            st.markdown("### 📝 Description")
            st.markdown(food_info.get("description", "N/A"))

            st.markdown("### 💪 Nutritional Highlights")
            st.markdown(f"**Calories:** {food_info.get('calories', 'N/A')}")
            st.markdown(f"**Macronutrients:** {food_info.get('macronutrients', 'N/A')}")
            st.markdown(f"**Micronutrients:** {food_info.get('micronutrients', 'N/A')}")
            st.markdown(f"**Nutritional Insights:** {food_info.get('nutritional_insights', 'N/A')}")
            st.markdown(f"**Health Risks:** ⚠️ {food_info.get('health_risks', 'N/A')}")

            st.markdown("### 🌍 Cultural Insights")
            st.markdown(food_info.get("cultural_insights", "N/A"))

            st.markdown("### 🍽️ Food Profile")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Category:** {food_info.get('category', 'N/A')}")
                st.markdown(f"**Diet Suitability:** {food_info.get('diet_suitability', 'N/A')}")
                st.markdown(f"**Seasonality:** {food_info.get('seasonality', 'N/A')}")
            with col2:
                st.markdown(f"**Common Allergens:** {food_info.get('common_allergens', 'N/A')}")
                st.markdown(f"**Interesting Fact:** {food_info.get('interesting_facts', 'N/A')}")

        else:
            st.error(f"❌ Server error: {response.text}")

    except Exception as e:
        st.error(f"❌ Failed to connect to API: {str(e)}")
