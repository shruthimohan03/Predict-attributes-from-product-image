# Predict attributes from product image
Predicting attributes from supplier uploaded images as part of cataloging is extremely crucial for any E-commerce platform. Typically, when suppliers upload their products for listing on the marketplace, they are required to fill in various details corresponding to product attributes (e.g., color, pattern, sleeve). This process often results in incorrect or incomplete attribute information.

## **Dataset Description**
1. **Training Dataset (`train.csv`)**  
   - Contains product information, including `Category` and 10 attribute columns (`attr_1` to `attr_10`).
   - Some attribute columns have missing values.
   - Example columns: `id`, `Category`, `len`, `attr_1`, ..., `attr_10`.

2. **Category Attributes (`category_attributes.parquet`)**  
   - Metadata on the number and type of attributes for each category.

3. **Testing Dataset (`test.csv`)**  
   - Similar to the training dataset but without attribute labels.

4. **Image Features (`train_features_26k.npy`, `test_features_30500.npy`)**  
   - Pre-extracted image features using a Vision Transformer (ViT).

---

## **Setup and Data Preparation**
1. **Install Dependencies**  
   Ensure you have the following libraries installed:
   - `pandas`, `numpy`, `tensorflow`, `scikit-learn`, `pyarrow`.

2. **Load and Inspect Data**
   - Import training, testing, and category attributes datasets.
   - Handle missing values by filling them with a placeholder (`'dummy_value'`).

3. **Filter for Specific Categories**
   - Product categories: "Men Tshirts", "Sarees", "Kurtis", "Women Tshirts" and "Women Tops & Tunics"

4. **Reshape Image Features**
   - Flatten the image features from `(32, 768)` to `(768)` per sample.

5. **Encode Attribute Labels**
   - Use `LabelEncoder` to encode attribute columns.
   - Convert the encoded labels into one-hot vectors for training.

---

## **Model Architecture**
The model combines **image features** and **tabular data** for predictions:
- **Input Layers:**
  - Image Features: `(768,)`
  - Tabular Data: `(2,)`
- **Hidden Layers:**
  - Dense layers for image and tabular data separately.
  - Concatenation of image and tabular layers.
- **Output Layers:**
  - Separate outputs for each attribute with a softmax activation.

---

## **Training the Model**
1. **Compile the Model**
   - Loss: Categorical Crossentropy for each attribute.
   - Metrics: Accuracy for each attribute.

2. **Fit the Model**
   - Input: Combined image and tabular data.
   - Output: One-hot encoded labels for all attributes.
   - Validation split: 20%
   - Epochs: 50  
   - Batch Size: 32  

---

## **Testing and Predictions**
1. **Prepare Test Features**
   - Filter test samples for "Men Tshirts"/other product categories
   - Reshape the image features as required.

2. **Get Predictions**
   - Use the trained model to predict attributes.
   - Extract the class with the highest probability for each attribute.

3. **Reverse Encoding**
   - Use `LabelEncoder.inverse_transform` to convert predicted labels back to original values.

4. **Submission File**
   - Create a DataFrame with predictions for all attributes.
   - Ensure proper alignment with test data indices.
