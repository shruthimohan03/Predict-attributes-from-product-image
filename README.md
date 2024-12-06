# Predict attributes from product image
Predicting attributes from supplier uploaded images as part of cataloging is extremely crucial for any E-commerce platform. Typically, when suppliers upload their products for listing on the marketplace, they are required to fill in various details corresponding to product attributes (e.g., color, pattern, sleeve). This process often results in incorrect or incomplete attribute information.

Dataset
1. Product ID
Unique identifier for each product.
2, Product Image
URLs of images that are publicly available online or can be accessed using productid.jpg.
3. Category
4. Attribute Keys
The attribute name specified for the product.
5. Attribute Values
The attribute values specified for the product corresponding to the attribute key.

Evaluation
We are going to evaluate the attribute classification model on a product dataset of the same categories. The attribute identification problem is posed as a multi-class classification problem at the attribute level. In our setting, an attribute can only take one value and hence it’s not a multi-label problem. To evaluate the model fairly, we will use micro and macro f1-score at attribute level. To calculate the f1 score at the category level, we take the unweighted average across all attribute f1 scores for that category.
