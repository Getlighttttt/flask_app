from flask import Flask, render_template, request, jsonify, Response
import torch
import pandas as pd
import MyKMeans as km
import matplotlib
matplotlib.use("Agg")  # Use a backend that does not require a GUI
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
IMG_DIR = os.path.join(STATIC_DIR, "images")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
IMG_DIR = os.path.join(STATIC_DIR, "images")

# Ensure both folders exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/cluster", methods=["POST"])
def cluster():
    dataset_url = request.form.get("url_get")  # Get from form data
    
    tol = request.form.get("tol")
    tol = int(tol) if tol and tol.isdigit() else 1000  # Ensure integer default
    
    try:
        df = pd.read_csv(dataset_url)
    except:
        return "<h1>Invalid url</h1>"
        
    df_numeric = df.select_dtypes(include=['number']).dropna()
    
    device = "cuda"
    X = torch.tensor(df_numeric.values, dtype=torch.float32).to(device)
    
    k = request.form.get("k")  # Get from form data
    k = int(k) if k and k.isdigit() else None
    if k != None:
        k = min(k, X.shape[0])
    
    opt, wcss_values, max_k, k = km.elbow_method(X, k, device=device, tol=tol)
    # Vykreslenie grafu Elbow Method
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, k + 1), wcss_values, marker='o', linestyle='-', color='b')
    ax.set_xlabel("Počet zhlukov (k)")
    ax.set_ylabel("Inertia (WCSS)")
    ax.set_title("Metóda lakťa (Elbow Method) pre výber optimálneho k")
    ax.set_xticks(range(1, max_k + 1))
    ax.grid()
    
    image_path = os.path.join(IMG_DIR, "plot.png")
    plt.savefig(image_path)
    plt.close(fig)


    return f"""
            <h1>Clustering started with k={k} and dataset URL={dataset_url}, tol={tol}</h1>
            <h1>{opt}, {wcss_values}</h1>
            <img src="/static/images/plot.png" alt="Elbow Method Graph">
            <br><br>
            <button onclick="window.location.href='/'">Go Back</button>
            """


if __name__ == "__main__":
    app.run(debug=True)