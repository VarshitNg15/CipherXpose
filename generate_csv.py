import pandas as pd
import os
import random
import string
from Crypto.Cipher import AES, DES, PKCS1_OAEP, ChaCha20
from Crypto.Util.Padding import pad
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from Crypto.Cipher import Blowfish
from Crypto.Cipher import DES3  # For 3DES
from Crypto.Cipher import AES, DES, ARC4, Salsa20
from Crypto.Protocol.KDF import PBKDF2
from base64 import b64encode

# Function to generate random hex strings
def generate_random_hex_string(length):
    return ''.join(random.choices(string.hexdigits, k=length))

# Function to encrypt using AES
def encrypt_aes(plaintext):
    key = get_random_bytes(16)  # AES key must be either 16, 24, or 32 bytes long
    cipher = AES.new(key, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(plaintext.encode(), AES.block_size))
    return cipher.iv.hex() + ":" + ct_bytes.hex()

# Function to encrypt using DES
def encrypt_des(plaintext):
    key = get_random_bytes(8)  # DES key must be exactly 8 bytes
    cipher = DES.new(key, DES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(plaintext.encode(), DES.block_size))
    return cipher.iv.hex() + ":" + ct_bytes.hex()

# Function to encrypt using RSA
def encrypt_rsa(plaintext):
    key = RSA.generate(2048)  # Generate RSA key pair
    public_key = key.publickey()
    cipher = PKCS1_OAEP.new(public_key)  # Create a cipher object using the public key
    ciphertext = cipher.encrypt(plaintext.encode())  # Encrypt the plaintext
    return ciphertext.hex(), key.export_key(format='PEM').decode()  # Return ciphertext and private key for later decryption

# Function to encrypt using ChaCha20
def encrypt_chacha20(plaintext):
    key = get_random_bytes(32)  # ChaCha20 key must be 32 bytes
    cipher = ChaCha20.new(key=key)
    ciphertext = cipher.encrypt(plaintext.encode())
    return cipher.nonce.hex() + ":" + ciphertext.hex()

# Function to encrypt using Blowfish
def encrypt_blowfish(plaintext):
    key = get_random_bytes(16)  # Blowfish key can be 4 to 56 bytes
    cipher = Blowfish.new(key, Blowfish.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(plaintext.encode(), Blowfish.block_size))
    return cipher.iv.hex() + ":" + ct_bytes.hex()

# Function to encrypt using 3DES
def encrypt_3des(plaintext):
    key = get_random_bytes(24)  # 3DES key must be 24 bytes
    cipher = DES3.new(key, DES3.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(plaintext.encode(), DES3.block_size))
    return cipher.iv.hex() + ":" + ct_bytes.hex()

# List of cryptography algorithms
algorithms = {
    'AES': encrypt_aes,
    'DES': encrypt_des,
    'RSA': encrypt_rsa,
    'ChaCha20': encrypt_chacha20,
    'Blowfish': encrypt_blowfish,
    '3DES': encrypt_3des,
}

# Generate dataset
data = []
num_samples = 10000  # Number of samples to generate
for _ in range(num_samples):
    algorithm = random.choice(list(algorithms.keys()))
    plaintext = generate_random_hex_string(32)  # Generate a random 32-character hex string
    
    if algorithm == 'RSA':
        ciphertext, private_key = algorithms[algorithm](plaintext)
        data.append({
            'plaintext': plaintext,
            'ciphertext': ciphertext,
            'algorithm': algorithm,
            'private_key': private_key  # Store the private key for RSA
        })
    else:
        ciphertext = algorithms[algorithm](plaintext)
        data.append({
            'plaintext': plaintext,
            'ciphertext': ciphertext,
            'algorithm': algorithm
        })

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV file
csv_file_name = "cryptography_dataset.csv"
df.to_csv(csv_file_name, index=False)

print(f"Dataset generated and saved to {csv_file_name} with {num_samples} samples.")
