import pandas as pd
import numpy as np

def generate_data(n=1000):
    np.random.seed(42)

    data = pd.DataFrame({
        "age": np.random.randint(22, 60, n),
        "salary": np.random.randint(20000, 150000, n),
        "department": np.random.choice(["HR", "IT", "Sales", "Finance"], n),
        "manager_score": np.random.randint(1, 10, n),
        "projects_completed": np.random.randint(0, 20, n),
        "experience": np.random.randint(1, 20, n)
    })

    # Target logic
    score = (
        data["manager_score"] * 2 +
        data["projects_completed"] * 0.5 +
        data["experience"] * 0.3 +
        (data["salary"] / 10000)
    )

    conditions = [
        score < 15,
        (score >= 15) & (score < 25),
        score >= 25
    ]

    choices = ["Low", "Medium", "High"]
    data["performance"] = np.select(conditions, choices)

    return data


if __name__ == "__main__":
    df = generate_data()
    df.to_csv("data/employee_data.csv", index=False)
    print("✅ Data generated successfully")