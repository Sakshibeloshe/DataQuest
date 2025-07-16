import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Simulate data for 50 startups
np.random.seed(42)
n_samples = 50

data = {
    'past_funding_rounds': np.random.randint(0, 5, n_samples),
    'time_between_rounds': np.random.randint(3, 36, n_samples),
    'traffic_growth': np.random.uniform(-0.5, 2.0, n_samples),  # -50% to +200%
    'revenue_million': np.random.uniform(0.5, 100, n_samples),
    'investor_quality': np.random.randint(1, 11, n_samples),  # 1 to 10

    # Target: 1 if expected to raise again, 0 otherwise
    'will_raise': np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
}

df = pd.DataFrame(data)

# Features and target
X = df[['past_funding_rounds', 'time_between_rounds', 'traffic_growth', 'revenue_million', 'investor_quality']]
y = df['will_raise']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Evaluation:")
print(classification_report(y_test, y_pred))

# Predict for a new startup (e.g., Palla)
new_startup = pd.DataFrame({
    'past_funding_rounds': [1],           # Series A = 1
    'time_between_rounds': [26],          # Months since last round
    'traffic_growth': [0.203],            # 20.3% growth
    'revenue_million': [10],              # $10M
    'investor_quality': [8]               # High (e.g., Y Combinator, Revolution Ventures)
})

prediction = model.predict(new_startup)
probability = model.predict_proba(new_startup)[0][1]

print(f"\nPrediction for Palla:")
print(f"Will likely raise another round? {'Yes' if prediction[0] == 1 else 'No'}")
print(f"Probability: {probability:.2%}")
