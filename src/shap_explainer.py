import shap

def explain_model(model, X):
    explainer = shap.Explainer(model.named_steps["model"])
    shap_values = explainer(X)
    return shap_values