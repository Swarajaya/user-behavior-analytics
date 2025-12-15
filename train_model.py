import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression

# Sample training data (for demo)
data = pd.DataFrame({
    "sessions": [1, 2, 3, 5, 8, 13, 21],
    "pageviews": [5, 10, 15, 30, 50, 80, 130],
    "timeOnSite": [50, 120, 300, 600, 1200, 2000, 3500],
    "converted": [0, 0, 0, 1, 1, 1, 1]
})

X = data[["sessions", "pageviews", "timeOnSite"]]
y = data["converted"]

model = LogisticRegression()
model.fit(X, y)

os.makedirs("model", exist_ok=True)
with open("model/conversion_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Real ML model saved")
