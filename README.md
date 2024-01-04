This is the fork of https://github.com/3DTopia/OpenLRM.git
# Contributions
- removing background and centering any input image using [Rembg](https://github.com/danielgatis/rembg)
- Python Notebook [demo.ipynb] (https://github.com/violetdenim/OpenLRM/blob/main/demo.ipynb)
- Gradio Application with model viewer [app.py] (https://github.com/violetdenim/OpenLRM/blob/main/app.py)


# OpenLRM: Open-Source Large Reconstruction Models

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-yellow.svg)](LICENSE)
[![Weight License](https://img.shields.io/badge/Weight%20License-CC%20By%20NC%204.0-red)](LICENSE_WEIGHT)
[![LRM](https://img.shields.io/badge/LRM-Arxiv%20Link-green)](https://arxiv.org/abs/2311.04400)

[![HF Models](https://img.shields.io/badge/Models-Huggingface%20Models-bron)](https://huggingface.co/zxhezexin/OpenLRM)
[![HF Demo](https://img.shields.io/badge/Demo-Huggingface%20Demo-blue)](https://huggingface.co/spaces/zxhezexin/OpenLRM)

# violetdenim Contribution
[![Google Colab](https://img.shields.io/badge/google-colab-red)](https://colab.research.google.com/drive/1qk4d6l9iG67h3AO2_iIsRQnKYFtpXGQs?usp=sharing0)

## Setup

### Installation
```
git clone https://github.com/violetdenim/OpenLRM.git
cd OpenLRM
```

### Environment
```
pip install -r requirements.txt
```

## To run Python Notebook run
```
jupyter notebook demo.ipynb
```
## To run gradio run
```
python app.py
```

### Images
- Added new inputs without background into `assets/sample_input`
- Added folder `assets/with_background` with source background

## License

- OpenLRM as a whole is licensed under the [Apache License, Version 2.0](LICENSE), while certain components are covered by [NVIDIA's proprietary license](LICENSE_NVIDIA). Users are responsible for complying with the respective licensing terms of each component.
- Model weights are licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE_WEIGHT). They are provided for research purposes only, and CANNOT be used commercially.
