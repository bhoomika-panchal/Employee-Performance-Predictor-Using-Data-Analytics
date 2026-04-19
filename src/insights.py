def generate_insight(prediction):

    if prediction == "Low":
        return "⚠️ Employee performance is LOW. Recommend training, mentoring, and closer supervision."

    elif prediction == "Medium":
        return "📈 Employee performance is MEDIUM. Suggest skill improvement and goal setting."

    else:
        return "🚀 Employee performance is HIGH. Eligible for promotion and leadership roles."