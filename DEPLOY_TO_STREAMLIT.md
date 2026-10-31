# Update the existing GitHub repository and Streamlit app

The previous Streamlit logs confirmed that package installation completed and the app server started, but they did not include a Python traceback. This revision changes the app to load the compact `GSE124326_preprocessed.npz` file by default instead of parsing the full 66 MB text matrix on every start.

## 1. Copy/update these files in your local `bipolar_ml_dashboard` folder
- `app.py`
- `requirements.txt`
- `GSE124326_preprocessed.npz`
- `five_fold_cv_training_results.csv`
- `README.md`

Keep `sample_metadata.csv`. The raw `GSE124326_count_matrix.txt` may remain in the repository for provenance, but the dashboard no longer parses it on app startup.

## 2. Push changes to your existing GitHub repository
Open Terminal in the folder containing `app.py` and run:
```bash
git add app.py requirements.txt GSE124326_preprocessed.npz five_fold_cv_training_results.csv README.md
git commit -m "Optimize dashboard data loading and evaluation"
git push
```

## 3. Let Streamlit redeploy
Open the deployed app. Streamlit Community Cloud should detect the GitHub commit and rebuild. If it does not, use the app menu's Reboot app option once.

## 4. Test
Open the public app URL. The dataset overview should appear automatically without uploading files. Click `Run / refresh analysis`; the app will run the three model pipelines and five-fold cross-validation.

The displayed predictions are exploratory and not for clinical use.
