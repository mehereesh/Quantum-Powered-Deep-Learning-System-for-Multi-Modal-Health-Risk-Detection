# Quantum-Powered-Deep-Learning-System-for-Multi-Modal-Health-Risk-Detection

In today’s fast-evolving world of healthcare, early and accurate detection of diseases can save lives. Our project focuses on creating a comprehensive health risk detection system that combines quantum computing techniques with deep learning algorithms to analyze and classify a wide range of health conditions.

The system processes different types of medical data—including brain MRI scans, ECG signals, chest X-rays, spinal cord MRIs, and basic vitals like heart rate and oxygen levels. By doing so, it provides an all-in-one platform for identifying diseases such as brain tumors, cardiac issues, respiratory disorders, and neurological conditions.

What sets this system apart is its speed and efficiency. Traditional machine learning methods often take a lot of time to process and can fall short in accuracy. In contrast, our system has shown to classify hundreds of images in a matter of seconds while maintaining high reliability. By leveraging the strengths of quantum-inspired processing and deep learning models, we aim to build a medical diagnostic tool that is not only powerful but also practical for real-world use.



for brain tumor:

Summary Table:

Purpose	Library	Usage in Notebook
Deep Learning	TensorFlow, Keras	Model building, training, evaluation
Machine Learning	scikit-learn	Train/test split, classification metrics
Quantum Technology	PennyLane	Quantum circuit simulation, quantum convolution layer
Image Processing	OpenCV, Matplotlib	Image resizing, visualization
Data Handling	Pandas, h5py	Loading data, labels, HDF5 file interaction
Numerical Ops	NumPy	Array and math operations (via PennyLane's numpy)



              precision    recall  f1-score   support

         1.0       0.93      0.84      0.88       113
         2.0       0.92      0.97      0.94       213
         3.0       0.98      0.97      0.97       132

    accuracy                           0.94       458
   macro avg       0.94      0.93      0.93       458
weighted avg       0.94      0.94      0.94       458
Common Python Machine Learning Libraries
scikit-learn (sklearn): For classic ML algorithms.
pandas, numpy: For data manipulation and numerical computation.
Deep Learning Libraries
tensorflow or keras
EfficientNetB0
matplotlib, seaborn: For plotting and visualization.
Quantum Technology Libraries
qiskit: IBM’s open-source quantum computing SDK.
pennylane: Quantum machine learning library.
cirq: Google’s quantum framework.
Other Potential Libraries
opencv-python: For image processing (since the dataset is ECG images).
albumentations, imgaug: For image augmentation.


91/91 [==============================] - 173s 2s/step
Classification Report (255x255 images):

              precision    recall  f1-score   support

           0       1.00      1.00      1.00       814
           1       1.00      1.00      1.00       716
           2       1.00      1.00      1.00       852
           3       0.99      1.00      1.00       516

    accuracy                           1.00      2898
   macro avg       1.00      1.00      1.00      2898
weighted avg       1.00      1.00      1.00      2898




heart ecg

