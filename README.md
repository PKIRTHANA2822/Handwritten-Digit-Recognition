# Project Title
- Handwritten Digit Recognition with Logistic Regression (scikit-learn Digits)
![hqdefault](https://github.com/user-attachments/assets/a5729bfb-f11c-4795-a5c8-5c338b758cce)
# Objective
- Build a classifier that predicts digits (0–9) from 8×8 grayscale images using Logistic Regression; evaluate with accuracy, classification report, and confusion matrix.
# Why This Project?
- Classic ML baseline for image classification before trying CNNs.
- Small, fast, and interpretable: great for teaching the full ML workflow (EDA → preprocessing → modeling → evaluation).
- Real uses: digitizing forms, postal codes, bank checks, meter readings.
# Step-by-Step Approach
- Load data: load_digits() → data.shape == (1797, 8, 8), target.shape == (1797,).
- Quick look: plot a 4×4 grid of sample images with labels.
- Reshape: images → flat vectors: (1797, 64).
- Scale: MinMaxScaler() to [0,1] (helps LR optimization).
- Split: train_test_split(test_size=0.25, random_state=42) → X_train (1347, 64), X_test (450, 64).
- Train model: LogisticRegression(...).
- Predict on test set.
- Evaluate: classification_report, accuracy_score, confusion matrix heatmap.
- Error analysis & iterate: inspect misclassified examples, tune hyperparameters.
# Exploratory Data Analysis (EDA)
- Sample visualization (already in your code): sanity-check labels vs images.
- Class balance: digits are roughly balanced (~180 per class).
- Intensity stats: pixels range 0–16 in raw data; after MinMax → [0,1].
- Average digit per class (optional): mean image for each label to see strokes.
- 2D view (optional): PCA/t-SNE of features to visualize separability.
# Feature Selection
- You can keep all 64 pixels, or try:
- VarianceThreshold to drop always-zero/low-variance pixels (often borders).
- SelectKBest (mutual_info_classif / chi²) to pick top K informative pixels.
- L1-regularized LR (penalty='l1', solver='saga') to induce sparsity.
- PCA (e.g., 30 components) for compact, denoised features.
# Feature Engineering
- Flattening 8×8 → 64-D vectors (done).
- Scaling with MinMax (done).
- (Optional) HOG or deskewing to capture stroke orientation/normalize slant.
- (Optional) PCA before LR for speed and robustness.
# Model Training
- Use Logistic Regression for multinomial classification.
- Practical, stable setup:
- lg = LogisticRegression(
    solver='lbfgs',        # good default for multinomial
    multi_class='auto',
    max_iter=1000,         # avoids convergence warnings
    C=1.0                  # tune with GridSearchCV
)
- lg.fit(X_train, y_train)
- Consider hyperparameter tuning: C (e.g., [0.01, 0.1, 1, 10]), with stratified CV.
# Model Testing
- Metrics: overall accuracy, and per-class precision/recall/F1 via classification_report.
- Confusion matrix: highlights common confusions (often 8↔9, 3↔5, 4↔9).
- Actual vs Predicted table: quick scan for systematic errors.
# Output (from your run)
<img width="218" height="759" alt="Screenshot 2025-08-16 193117" src="https://github.com/user-attachments/assets/1bf78895-bb49-42b4-b077-e83e389a8fdf" />
<img width="220" height="827" alt="Screenshot 2025-08-16 193132" src="https://github.com/user-attachments/assets/37883b18-ab23-4343-8cd6-cb65587ce8b9" />
<img width="216" height="132" alt="Screenshot 2025-08-16 193147" src="https://github.com/user-attachments/assets/5d42d21f-8ce9-4d49-a49b-3391f3d24ce8" />

