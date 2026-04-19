# 🌍 EcoShift  
### Predicting Biodiversity Under Climate Warming

**DataHacks 2026 · AI/ML Track · Climate & Environment**

EcoShift is an AI-driven climate–biodiversity visualization that predicts how rising temperatures reshape species distributions using real environmental and ecological data.

---

## 🚀 Overview

Climate change is not abstract — it is actively reshaping ecosystems.

EcoShift makes this visible.

We combine **real UCSD sensor climate data** with **research-grade iNaturalist species observations** to train a machine learning model that:

- Learns how environmental conditions affect species presence  
- Predicts biodiversity at any location and time  
- Simulates warming scenarios (+1°C to +5°C) instantly  
- Visualizes ecological shifts in an interactive map  

👉 The result: a **local, real-time view of how climate change impacts biodiversity**

---

## 🧠 Key Idea

We train a model to learn:

> “Given temperature, humidity, location, and season — which species are likely here?”

Then we ask:

> “What happens if temperature increases by +2°C?”

No retraining. Just a new input.

This enables **counterfactual climate simulation**.

---

## 📊 Data

### 🌡 UCSD Heat Map (Scripps)
- Mobile weather stations across UCSD campus  
- Temperature (°C), humidity (%)  
- Precise location + timestamp  
- Captures real microclimate variation  

### 🐦 iNaturalist (Research-grade)
- Verified species observations  
- Species label, location, time  
- Represents real biodiversity  

### 🔗 Data Fusion
- Matched using spatial + temporal proximity:
  - within **500 meters**
  - within **2 hours**
- Efficiently computed using KD-tree  

📌 **Suggested image:**
- Heatmap visualization of UCSD sensor data  
- Raw iNaturalist point scatter  

---

## ⚙️ Machine Learning Pipeline

### 1. Ingest
Join sensor data and species observations using spatial-temporal matching  

### 2. Bias Correction
Fix observation bias:
- Spatial thinning (reduce over-sampled regions)  
- Target-group background sampling (MaxEnt-style)  

### 3. Feature Engineering
Each data point becomes 6 features:
- Temperature  
- Humidity  
- Latitude  
- Longitude  
- sin(day-of-year), cos(day-of-year)  

### 4. Model
Multi-label neural network (MLP):


6 → 128 → 64 → 32 → Species Outputs


- Predicts probability for all species simultaneously  
- Uses:
  - ReLU  
  - BatchNorm  
  - Dropout (0.3)  
  - BCEWithLogitsLoss  
  - Per-species class weighting  

### 5. Evaluation
Spatial cross-validation (5×5 geographic blocks):
- Prevents spatial data leakage  
- Ensures real generalization  

Low-performing species (AUC < 0.65) are flagged as low-confidence  

### 6. Inference
- Predict species probabilities at any location  
- Generate spatial grids  
- Apply warming offsets for simulation  

📌 **Suggested diagram:**
- Pipeline flow diagram (ingest → model → prediction)

---

## 🔮 Counterfactual Simulation

We simulate warming by modifying input temperature:

```python
temperature_c + offset
+1°C, +2°C, +3°C scenarios
No retraining required
All other variables held constant

👉 This answers:

“What would biodiversity look like if it were warmer?”

📌 Suggested visual:

Before vs After heatmap comparison (0°C vs +2°C)
🗺 Visualization

Interactive frontend includes:

Real heat/climatology layer (UCSD microclimate)
Species suitability layer (model predictions)
Warming slider
Hover + click inspection
Confidence indicators

📌 Suggested visuals:

Screenshot of map UI
GIF of warming slider changing predictions
Hover tooltip example
🛠 Tech Stack

Backend

Python
PyTorch
NumPy / Pandas / Scikit-learn
FastAPI

Frontend

React / Next.js
Leaflet / Mapbox / Deck.gl
▶️ Running Locally
1. Clone repo
git clone https://github.com/dhruvsood12/Datahacks-2026.git
cd Datahacks-2026
2. Setup environment
python3 -m venv venv
source venv/bin/activate
3. Install dependencies
pip install torch numpy pandas scikit-learn matplotlib seaborn pyarrow tqdm joblib fastapi uvicorn
4. Test model
python -c "from pipeline.inference import Predictor; p=Predictor.load(); print(p.health())"
5. Run backend
uvicorn api.main:app --reload
6. Run frontend
cd frontend
npm install
npm run dev
🎯 Why This Matters

EcoShift turns climate change into something you can see:

Identify climate winners (expanding species)
Identify climate losers (declining species)
Understand local ecological impact
Enable data-driven conservation insights
🔭 Future Work
Expand beyond UCSD → city / regional scale
Add more environmental variables (precipitation, vegetation, elevation)
Improve model accuracy with larger datasets
Add uncertainty visualization
Deploy as a real-world ecological monitoring tool
🏆 DataHacks 2026 Submission
Track: AI / ML · Climate & Environment
Bonus: Scripps Challenge (uses UCSD Heat Map + iNaturalist)
👥 Team

Built for DataHacks 2026
UC San Diego
