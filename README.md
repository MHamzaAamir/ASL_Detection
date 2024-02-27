# ASL_Detection

**MAIN LIBRARIES USED:**
Media Pipe, Scikit Learn, OpenCV

**Description:**
In this project, I have developed an American Sign Language (ASL) detection system using MediaPipe for hand landmark extraction and scikit-learn's Random Forest model for training on the extracted coordinates. The dataset, meticulously curated for each letter of the alphabet, comprises 100 photos per letter, ensuring diverse and comprehensive training samples. MediaPipe's hand landmark extraction provides a robust representation of hand gestures, and the Random Forest model, implemented through scikit-learn, effectively learns the spatial relationships among the landmarks. The combination of these technologies empowers the system to accurately interpret and predict ASL gestures, making it a valuable tool for enhancing accessibility and communication for individuals with hearing impairments.Furthermore, the project seamlessly integrates with OpenCV to enable real-time ASL detection using a webcam. By leveraging the capabilities of MediaPipe and scikit-learn's Random Forest model, the system processes live video feed from the webcam, extracting hand landmarks and making predictions on the detected gestures. The integration with OpenCV not only facilitates the practical application of the ASL detection model in real-world scenarios but also opens avenues for interactive and dynamic user experience.
