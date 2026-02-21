
import json
import os

nb_path = r"d:\Projects\Retail_Customer_Intelligence\notebooks\04_churn_modeling.ipynb"

print(f"Reading notebook from: {nb_path}")

shap_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Feature Importance & Explainability\n",
            "\n",
            "Understanding which features drive churn predictions using SHAP values and Permutation Importance."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Permutation Importance\n",
            "from sklearn.inspection import permutation_importance\n",
            "\n",
            "print(\"Calculating Permutation Importance...\")\n",
            "result = permutation_importance(\n",
            "    rf, X_val[num_cols], y_val, n_repeats=10, random_state=42, n_jobs=-1\n",
            ")\n",
            "\n",
            "perm_sorted_idx = result.importances_mean.argsort()\n",
            "\n",
            "plt.figure(figsize=(10, 6))\n",
            "plt.boxplot(\n",
            "    result.importances[perm_sorted_idx].T,\n",
            "    vert=False,\n",
            "    labels=X_val[num_cols].columns[perm_sorted_idx],\n",
            ")\n",
            "plt.title(\"Permutation Importance (Validation Set)\")\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# SHAP Analysis\n",
            "try:\n",
            "    import shap\n",
            "    \n",
            "    print(\"Calculating SHAP values...\")\n",
            "    # Use a sample for speed\n",
            "    X_val_sample = X_val[num_cols].sample(n=500, random_state=42) if len(X_val) > 500 else X_val[num_cols]\n",
            "    \n",
            "    # TreeExplainer for Random Forest\n",
            "    explainer = shap.TreeExplainer(rf)\n",
            "    # shap_values is a list for classifiers [class0, class1]\n",
            "    shap_values = explainer.shap_values(X_val_sample)\n",
            "    \n",
            "    # Summary Plot (Bar)\n",
            "    plt.figure(figsize=(10, 6))\n",
            "    shap.summary_plot(shap_values[1], X_val_sample, plot_type=\"bar\", show=False)\n",
            "    plt.title(\"SHAP Feature Importance (Global)\")\n",
            "    plt.show()\n",
            "    \n",
            "    # Summary Plot (Dot)\n",
            "    plt.figure(figsize=(10, 6))\n",
            "    shap.summary_plot(shap_values[1], X_val_sample, show=False)\n",
            "    plt.title(\"SHAP Summary Plot (Impact on Churn Risk)\")\n",
            "    plt.show()\n",
            "    \n",
            "except ImportError:\n",
            "    print(\"SHAP library not found. Please install with: pip install shap\")\n",
            "except Exception as e:\n",
            "    print(f\"Error calculating SHAP values: {e\")"
        ]
    }
]

try:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Find where to insert (before "Save Model Artifacts")
    insert_idx = len(nb["cells"])
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "markdown" and "Save Model Artifacts" in "".join(cell["source"]):
            insert_idx = i
            break
    
    print(f"Inserting {len(shap_cells)} cells at index {insert_idx}")
    
    # Check if already inserted to avoid duplicates
    if insert_idx > 0 and "Feature Importance" in "".join(nb["cells"][insert_idx-1]["source"]):
         print("SHAP cells seem to be already present. Skipping.")
    else:
        for cell in reversed(shap_cells):
            nb["cells"].insert(insert_idx, cell)
        
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print("Notebook updated with SHAP analysis.")

except Exception as e:
    print(f"Error: {e}")
