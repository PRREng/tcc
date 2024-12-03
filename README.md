# EIS Deep Learning Classifier

A deep learning-based model for **Electrochemical Impedance Spectroscopy (EIS)** analysis, designed to classify equivalent circuits. This tool aims to assist researchers and engineers in automating the identification of equivalent circuit models, streamlining the analysis of impedance data.

## Features
- Supports multiple equivalent circuit configurations:
  - **C1:** (R + R-CPE)
  - **C2:** (R + R-CPE + R-CPE)
  - **C3:** (R + CPE-(R + W))
  - **C4:** (R + R-CPE + CPE-(R + W))
  - **C5:** (R + CPE-(R + CPE-(R + W)))
- Processes impedance data as sequences, leveraging Transformer-based architectures.
- Handles missing or infinite values in datasets.
- Multi-class classification with support for up to 5 classes.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/PRREng/tcc.git
   cd tcc
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure your data is formatted correctly and preprocessed (see `data/README.md` for details).

## Usage

### Training
To train the model:
```bash
python expandai.py -m hybridv4 -l true
```

### Evaluation
To evaluate the model on test data:
```bash
python testmodel.py
```

## License

This project is licensed under the MIT License. See `LICENSE` for more details.

## Contact

For questions or suggestions, feel free to reach out via [email](mailto:your-email@example.com).
```

Let me know if you'd like me to adjust anything, such as placeholders or add additional sections!
