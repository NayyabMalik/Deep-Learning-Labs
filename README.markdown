# Deep Learning Labs

This repository contains my deep learning coursework from the National University of Modern Languages, Islamabad, submitted on October 31, 2024, as part of the Faculty of Engineering & Computer Science. It includes lab manuals (summarized for educational purposes) and practical implementations in Jupyter notebooks, covering key deep learning concepts and their applications in image classification, time series forecasting, and object detection.

## Lab Manuals
The `lab_manuals/` folder contains summarized versions of the lab manuals from my deep learning course, authored by Nayyab Malik (BSAI-127) under the supervision of Mam Iqra Nasem. These manuals provide theoretical foundations for the practical projects in the `notebooks/` folder. **Note**: The content is shared with permission for educational purposes only and has been redacted to exclude proprietary or sensitive information.

| Lab | Manual Title | Description | Related Notebook |
|-----|--------------|-------------|-----------------|
| Lab 1 | ANN_Classification.pdf | Introduces classification using Artificial Neural Networks (ANNs), covering gradient descent, learning rates, hidden units, and activation functions (ReLU, Tanh, Sigmoid). | alzheimers_prediction.ipynb |
| Lab 2 | Loss_Functions.pdf | Explores loss functions (MSE, MAE, Binary Cross-Entropy) and their properties for regression and classification tasks. | alzheimers_prediction.ipynb |
| Lab 3 | CNN_Classification.pdf | Covers Convolutional Neural Networks (CNNs) for classification, including data preprocessing, convolutional layers, and pooling. | Pneumonia_detection_using_X-rays.ipynb |
| Lab 4 | CNN_Patterns.pdf | Discusses CNN preprocessing (e.g., grayscale image loading, edge detection) and architecture components. | Pneumonia_detection_using_X-rays.ipynb |
| Lab 4+5 | CNN_Advanced.pdf | Details advanced CNN architecture with weight initialization and regularization techniques. | Pneumonia_detection_using_X-rays.ipynb |
| Lab 6+7 | VGG16_MNIST.pdf | Explores transfer learning with VGG16 on the MNIST dataset, including data augmentation and fine-tuning. | None (MNIST-specific) |
| Lab 6+7 | ResNet_CatDog.pdf | Covers binary classification using ResNet50 for cat vs. dog images. | None (Cat/Dog-specific) |
| Lab 8+9 | Bounding_Boxes.pdf | Introduces object detection with bounding box prediction on MNIST using convolutional layers. | None (MNIST-specific) |
| Lab 10 | AutoEncoders.pdf | Implements a single-layer autoencoder for MNIST image compression and reconstruction. | None (MNIST-specific) |
| Lab 11 | RNN_Text.pdf | Describes Recurrent Neural Networks (RNNs) for text generation, including data preprocessing and sequence modeling. | anomaly_detection.ipynb |
| Lab 12+13 | LSTM_TimeSeries.pdf | Covers Long Short-Term Memory (LSTM) networks for time series forecasting with synthetic data. | anomaly_detection.ipynb |

## Notebooks
The `notebooks/` folder contains practical implementations of deep learning models, applying concepts from the lab manuals:

- **Pneumonia_detection_using_X-rays.ipynb**: A CNN model for classifying chest X-ray images as Normal or Pneumonia, using the Kaggle Chest X-Ray dataset. Links to Lab 3, 4, and 4+5 for CNN theory and preprocessing.
- **anomaly_detection.ipynb**: Implements CNN, CNN+RNN, and CNN+LSTM models for real-time anomaly detection in CCTV footage (UCF-Crime dataset). Links to Lab 11 (RNN) and Lab 12+13 (LSTM) for sequence modeling.
- **alzheimers_prediction.ipynb**: Uses machine learning models (e.g., ANN, Random Forest) for Alzheimer’s disease prediction. Links to Lab 1 (ANN) and Lab 2 (Loss Functions) for classification theory.

## Repository Structure
```
deep-learning-labs/
├── lab_manuals/
│   ├── ANN_Classification.pdf
│   ├── Loss_Functions.pdf
│   ├── CNN_Classification.pdf
│   ├── CNN_Patterns.pdf
│   ├── CNN_Advanced.pdf
│   ├── VGG16_MNIST.pdf
│   ├── ResNet_CatDog.pdf
│   ├── Bounding_Boxes.pdf
│   ├── AutoEncoders.pdf
│   ├── RNN_Text.pdf
│   ├── LSTM_TimeSeries.pdf
├── notebooks/
│   ├── Pneumonia_detection_using_X-rays.ipynb
│   ├── anomaly_detection.ipynb
│   ├── alzheimers_prediction.ipynb
├── results/
│   ├── (plots, models, or other outputs)
├── requirements.txt
├── README.md
├── LICENSE
```

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/deep-learning-labs.git
   cd deep-learning-labs
   ```

2. **Install Dependencies**:
   Install the required Python libraries listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   Key libraries include: `tensorflow`, `keras`, `numpy`, `opencv-python`, `matplotlib`, `scikit-learn`, `torch`.

3. **Download Datasets**:
   - **Chest X-Ray Dataset**: [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
   - **UCF-Crime Dataset**: [UCF-Crime Dataset](https://www.crcv.ucf.edu/data/UCF_Crime.php)
   - **Alzheimer’s Dataset**: [Kaggle Alzheimer’s Dataset](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images)
   - Place datasets in a `data/` folder or update notebook paths accordingly.

4. **Run Notebooks**:
   Open Jupyter Notebook and execute the desired notebook:
   ```bash
   jupyter notebook notebooks/<notebook_name>.ipynb
   ```

5. **View Lab Manuals**:
   The `lab_manuals/` folder contains PDF summaries of the theoretical content. Open them to understand the concepts behind the notebooks.

## Notes
- **File Size**: PDFs have been compressed to reduce repository size. Large files are managed using Git LFS (`git lfs track "*.pdf"`).
- **Permissions**: The lab manuals are summarized and shared with permission from the National University of Modern Languages for educational purposes. Contact the instructor (Mam Iqra Nasem) for access to original documents if needed.
- **Projects**: The notebooks focus on practical applications (pneumonia detection, anomaly detection, Alzheimer’s prediction), while the manuals provide theoretical context. Not all labs (e.g., MNIST, Cat/Dog) have corresponding notebooks in this repository but are included for completeness.

## Future Improvements
- Add visualizations for training/validation loss and accuracy in notebooks (e.g., for `Pneumonia_detection_using_X-rays.ipynb`, which shows overfitting).
- Implement data augmentation in `Pneumonia_detection_using_X-rays.ipynb` to improve validation accuracy (currently ~50%).
- Include precision, recall, and F1-score metrics for `alzheimers_prediction.ipynb`.
- Explore transfer learning (e.g., VGG16, ResNet) for `Pneumonia_detection_using_X-rays.ipynb` to enhance performance.

## License
This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.