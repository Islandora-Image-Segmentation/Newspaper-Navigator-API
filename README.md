# Newspaper-Navigator-API

An API wrapper around the Newspaper Navigator app.

## Getting Started

 1. Clone this repo.
 2. Download the pre-trained model weights from [here](https://drive.google.com/file/d/1qUu3uQ8imLGp-m4DYEY5KDCrNaaS2Pb-/view?usp=sharing).
 3. Place the model weights in the `resources/` folder.
 4. Install [PyTorch>=1.7.1](https://pytorch.org/).
 5. Install [detectron2.](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)
 6. Install the required Python packages from requirements.txt  
	 - You can run `pip install -r requirements.txt` to install all of them.
 7. Windows platform users may not have pytesseract in their PATH. If so add this line to your code `pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'`. For example, `tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'`