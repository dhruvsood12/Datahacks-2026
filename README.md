# 🌍 EcoShift  
### Predicting Biodiversity Under Climate Warming  

**DataHacks 2026 · AI/ML Track · Climate & Environment**

EcoShift is an AI-driven climate–biodiversity visualization that predicts how rising temperatures reshape species distributions using real environmental and ecological data.

---

## 🚀 Overview

Climate change is not abstract — it is actively reshaping ecosystems.

EcoShift makes this visible.

We combine real UCSD sensor climate data with research-grade iNaturalist species observations to train a machine learning model that:

- Learns how environmental conditions affect species presence  
- Predicts biodiversity at any location and time  
- Simulates warming scenarios (+1°C to +5°C) instantly  
- Visualizes ecological shifts in an interactive map  

The result is a local, real-time view of how climate change impacts biodiversity.

---

## 🧠 Key Idea

We train a model to learn:

“Given temperature, humidity, location, and season — which species are likely here?”

Then we ask:

“What happens if temperature increases by +2°C?”

No retraining. Just a new input.

This enables counterfactual climate simulation.

---

## 📊 Data

### UCSD Heat Map (Scripps)
- Mobile weather stations across UCSD campus  
- Temperature (°C), humidity (%)  
- Precise location and timestamp  
- Captures real microclimate variation  

### iNaturalist (Research-grade)
- Verified species observations  
- Species label, location, time  
- Represents real biodiversity  

### Data Fusion
- Matched using spatial + temporal proximity:
  - within 500 meters  
  - within 2 hours  
- Efficiently computed using KD-tree  

(Add image: heatmap + species scatter)

---

## ⚙️ Machine Learning Pipeline

1. Ingest  
   Match sensor data with species observations  

2. Bias Correction  
   - Spatial thinning  
   - Target-group background sampling (MaxEnt-style)  

3. Features  
   - Temperature  
   - Humidity  
   - Latitude, Longitude  
   - sin(day-of-year), cos(day-of-year)  

4. Model  
   Multi-label neural network:  
   6 → 128 → 64 → 32 → species outputs  

   Uses:
   - ReLU  
   - BatchNorm  
   - Dropout (0.3)  
   - BCEWithLogitsLoss  
   - Per-species class weighting  

5. Evaluation  
   - Spatial cross-validation (5×5 blocks)  
   - Prevents data leakage  
   - Ensures real generalization  

6. Inference  
   - Predict species at any location  
   - Generate spatial grids  
   - Apply warming offsets  

(Add pipeline diagram)

---

## 🔮 Counterfactual Simulation

We simulate warming by modifying temperature:

temperature_c + offset  

- +1°C, +2°C, +3°C scenarios  
- No retraining required  
- Other variables held constant  

This answers:

“What would biodiversity look like if it were warmer?”

(Add before/after map)

---

## 🗺 Visualization

Interactive frontend includes:

- Real heat/climatology layer (UCSD microclimate)  
- Species suitability layer (model predictions)  
- Warming slider  
- Hover and click inspection  
- Confidence indicators  

(Add UI screenshot + GIF)

---

## 🛠 Tech Stack

Backend:
- Python  
- PyTorch  
- NumPy, Pandas, Scikit-learn  
- FastAPI  

Frontend:
- React / Next.js  
- Leaflet / Mapbox / Deck.gl  

---

## ▶️ Running Locally

Clone repo:
git clone https://github.com/dhruvsood12/Datahacks-2026.git  
cd Datahacks-2026  

Setup environment:
python3 -m venv venv  
source venv/bin/activate  

Install dependencies:
pip install torch numpy pandas scikit-learn matplotlib seaborn pyarrow tqdm joblib fastapi uvicorn  

Test model:
python -c "from pipeline.inference import Predictor; p=Predictor.load(); print(p.health())"  

Run backend:
uvicorn api.main:app --reload  

Run frontend:
cd frontend  
npm install  
npm run dev  

---

## 🎯 Why This Matters

EcoShift turns climate change into something you can see:

- Identify climate winners (expanding species)  
- Identify climate losers (declining species)  
- Understand local ecological impact  
- Enable data-driven conservation insights  

---

## 🔭 Future Work

- Expand beyond UCSD to city-scale modeling  
- Add more environmental variables  
- Improve model accuracy with larger datasets  
- Add uncertainty visualization  
- Deploy as a real-world ecological monitoring tool  

---

## 🏆 DataHacks 2026 Submission

- Track: AI / ML · Climate & Environment  
- Bonus: Scripps Challenge (UCSD Heat Map + iNaturalist)  

---

## 👥 Team

Built for DataHacks 2026  
UC San Diego  

---

## 🌱 Final Thought

EcoShift provides a concrete, data-driven view of what climate change means for biodiversity — locally, visually, and in real time.
