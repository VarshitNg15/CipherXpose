# **CipherXPose: Cryptographic Algorithm Detection**
## **Identification of Encryption Algorithm Using AI/ML** 
CipherXPose is an AI/ML-powered system that detects cryptographic algorithms used in encryption and hashing processes. The project leverages deep learning techniques to analyze given ciphertexts and determine whether they belong to Block Ciphers, Stream Ciphers, or Hashing Algorithms.

## **Background**
With the growing importance of cryptography in secure communication, it becomes essential to identify and analyze encryption methods for security auditing, cryptanalysis, and forensic investigations. This project aims to enhance algorithm detection accuracy using machine learning models trained on structured datasets.

By accurately determining the cryptographic technique used, CipherXPose helps in:

Identifying algorithm weaknesses in implementations.

Auditing security compliance in encrypted communications.

Enhancing cybersecurity measures against cryptographic attacks.

## **Objective**
The primary goal of CipherXPose is to develop an intelligent system that can:

Detect whether a given ciphertext corresponds to a Block Cipher, Stream Cipher, or Hashing Algorithm.

Analyze cryptographic patterns using entropy, skewness, kurtosis, and unique byte distribution.

Identify potential weaknesses in the ciphertext, such as low entropy or byte distribution anomalies.

Provide a user-friendly interface for real-time predictions and analysis.

## **Solution Overview**
CipherXPose uses Convolutional Neural Networks (CNNs) and advanced feature engineering techniques to classify encryption algorithms. The solution includes:

AI-driven Cryptographic Algorithm Identification

Feature Extraction based on Statistical & Pattern Analysis

Weakness Detection (Entropy, Skewness, Kurtosis, and Bit Variability Analysis)

Interactive UI built with Gradio for easy usage

## **Cryptographic Techniques Covered**
CipherXPose detects and classifies three primary cryptographic algorithm types:

### **Block Ciphers**
Encrypt data in fixed-size blocks (e.g., AES, DES, 3DES). These algorithms ensure high security and are widely used in secure data transmission.
Modes of Operation:

ECB (Electronic Codebook) – Susceptible to pattern detection.

CBC (Cipher Block Chaining) – Stronger due to dependency on previous ciphertext blocks.

### **Stream Ciphers**
Encrypt data one bit or byte at a time, making them efficient for real-time data encryption (e.g., RC4, Salsa20). These ciphers are lightweight and widely used in wireless communication and secure VoIP systems.

### **Hashing Algorithms**
Transform data into a fixed-length hash value. Hash functions (e.g., SHA-256, MD5, SHA-1) are irreversible and used for password hashing, digital signatures, and data integrity verification.


## **Feature Analysis in Weakness Detection**
CipherXPose evaluates statistical properties of the ciphertext to detect weaknesses.

### **Feature	Ideal Range	Weakness Indicators**

Entropy	7.5 - 8.0	📉 Low entropy (< 3.0) → Weak randomness, vulnerable encryption

Skewness	~0	📈 High skewness (>
Kurtosis	~3	🛑 High kurtosis (> 3.0) → Possible repetitive patterns

Unique Byte Count	~256	⚠️ Low count (< 50) → Ciphertext predictability

## **Technologies Used**
Machine Learning Framework: TensorFlow, Scikit-learn

Programming Language: Python

Data Processing: NumPy, Pandas

Visualization: Matplotlib

UI Framework: Gradio

