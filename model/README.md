# Model Directory

Place your trained Keras model file inside this directory before running the Flask application.

Supported filenames:
- `trained_model.keras` (Recommended)
- `product_defect_cnn_model.h5`

### Notebook Export Instructions:
In your Jupyter Notebook, run:
```python
# Modern Keras format (Recommended)
model.save("model/trained_model.keras")

# Or HDF5 format
model.save("model/product_defect_cnn_model.h5")
```
