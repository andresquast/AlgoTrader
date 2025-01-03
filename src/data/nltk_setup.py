"""
NLTK setup script with SSL certificate handling
"""
import os
import ssl
import nltk
import subprocess

def setup_nltk():
    # Fix SSL
    try:
        subprocess.check_call(['/Applications/Python 3.x/Install Certificates.command'], shell=True)
    except:
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
    
    # Create NLTK data directory
    nltk_data_path = os.path.expanduser('~/nltk_data')
    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path)
    
    # Download required NLTK data
    required_packages = ['stopwords', 'punkt', 'averaged_perceptron_tagger', 'vader_lexicon']
    for package in required_packages:
        try:
            nltk.download(package, quiet=True)
        except Exception as e:
            print(f"Error downloading {package}: {str(e)}")

def main():
    setup_nltk()

if __name__ == '__main__':
    main()