# Multimodal Digit Recognition

This project implements a multimodal deep learning model for digit recognition using both image and audio data. The model combines a Convolutional Neural Network (CNN) for processing image data and a Neural Network for processing audio data.

## Features

- Data preprocessing and visualization
- Custom PyTorch Dataset for handling multimodal data
- CNN model for image processing
- Neural Network model for audio processing
- Combined model for multimodal digit recognition
- Training and evaluation pipeline
- Feature extraction and visualization using t-SNE and K-means clustering

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- Pandas
- Scikit-learn
- Seaborn

You can install the required packages using:

```
pip install torch torchvision numpy matplotlib pandas scikit-learn seaborn
```

## Usage

1. Prepare your data:
   - `x_train_wr.npy`: Training images
   - `x_train_sp.npy`: Training audio
   - `y_train.csv`: Training labels
   - `x_test_wr.npy`: Test images
   - `x_test_sp.npy`: Test audio

2. Run the script:

```
python multimodal_digit_recognition.py
```

3. The script will:
   - Load and preprocess the data
   - Visualize sample image and audio data
   - Train the multimodal model
   - Make predictions on the test set
   - Save predictions to `predictions.csv`
   - Visualize extracted features using t-SNE and K-means clustering

## Model Architecture

The model consists of three main components:

1. ImageCNN: A Convolutional Neural Network for processing image data
2. AudioNN: A Neural Network for processing audio data
3. CombinedModel: A model that combines the outputs of ImageCNN and AudioNN for final prediction

## Results

The model's performance can be evaluated based on the validation accuracy printed during training. The best model is saved as `best_model.pth`.
I was able to obtain 99.18% accuracy on the test set.

## Feature Visualization

The script includes functionality to visualize the extracted features using t-SNE dimensionality reduction and K-means clustering. This can provide insights into how well the model separates different digit classes in the feature space.

## Future Improvements

- Experiment with different model architectures
- Implement data augmentation techniques
- Try different hyperparameters for optimization
- Explore other feature visualization techniques

## License

This project is open-source and available under the MIT License.
