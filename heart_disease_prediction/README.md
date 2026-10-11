A Streamlit web app that predicts whether heart disease is present from four patient details:
age, gender, chest pain type and exercise-induced angina. The model is a StandardScaler + Support
Vector Classifier pipeline trained on the combined UCI heart disease data (865 patients, about 80% accurate on the test split).

**Educational use only. This is not a medical device and cannot diagnose or rule out heart disease.**

## Files
| File | Purpose |
| ---- | ------- |
| `app.py` | The Streamlit app |
| `svc_trained_model.pkl` | Trained model (created by Step 6.4 of the notebook) |
| `heart_disease_combined.csv` | Dataset. Used by the "Only offer patients from the dataset" switch |

## Run locally
```
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Live app
Add your link here after deploying: https://abdul-manan-4of6vkpqiwh7dncty6xyfn.streamlit.app/
