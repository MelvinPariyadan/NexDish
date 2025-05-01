from flask import Flask, jsonify
from openai import OpenAI
import os
import json
import time
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

API_KEY = os.getenv("OPENAI_API_KEY")

MODEL_NAME = "gpt-4o-mini-2024-07-18"
client = OpenAI(api_key=API_KEY)

def build_prompt(food_name):
    return f"""
        You are a friendly AI helping students learn about food!

        Write a short, light-hearted description for the food "{food_name}" using the sections below. Use a casual and friendly tone. Keep each section short—no more than 2–3 sentences if needed.

        Sections to include:
        1. description – Describe the food in an appetizing way.
        2. cultural_insights – Fun or casual cultural background (e.g., where it's popular, any traditions or symbolism).
        3. nutritional_insights – Informal notes on what it's made of and general nutrition (not too technical).
        4. calories – Estimated calorie range per serving (e.g., "250–300 kcal").
        5. seasonality – When it's in season or most commonly available.
        6. common_allergens – E.g., nuts, dairy, gluten.
        7. diet_suitability – E.g., vegan, vegetarian, gluten-free, halal.
        8. health_risks – Possible risks (e.g., "too much may cause bloating or spike blood sugar").
        9. macronutrients – Brief overview of carbs, protein, fats.
        10. micronutrients – Notable vitamins and minerals (e.g., Vitamin A, C, D, Iron, Calcium).
        11. interesting_facts – Surprising or fun trivia about the food.
        12. category – What type of food it is (e.g., fruit, vegetable, grain, meat).

        Return the result as a **valid JSON object** using the exact structure below:

        {{
        "description": "...",
        "cultural_insights": "...",
        "nutritional_insights": "...",
        "calories": "...",
        "seasonality": "...",
        "common_allergens": "...",
        "diet_suitability": "...",
        "health_risks": "...",
        "macronutrients": "...",
        "micronutrients": "...",
        "interesting_facts": "...",
        "category": "..."
        }}

        Do not include any extra commentary or formatting outside the JSON.
        """


def generate_insights(food_name, max_retries=2):
    prompt = build_prompt(food_name)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            insights = json.loads(content)
            return insights
        except Exception as e:
            print(f"Attempt {attempt+1} failed for '{food_name}': {e}")
    
    return {"error": f"Failed to generate insights for '{food_name}' after {max_retries} attempts."}

@app.route('/info/<food_label>', methods=['GET'])
def food_info(food_label):
    insights = generate_insights(food_label)
    return jsonify(insights)

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8001, debug=True)
