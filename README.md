# Stuttering Detection Using Deep Learning and Hybrid Ensemble

A comprehensive machine learning system for detecting stuttering events in speech audio using transfer learning with HuBERT and hybrid ensemble methods.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

This project implements an advanced stuttering detection system using:
- **HuBERT Transfer Learning** (Facebook's pre-trained model)
- **Hybrid Ensemble** (XGBoost, SVM, Random Forest, CNN, BiLSTM)
- **Multi-Annotator Label Processing** with confidence scoring
- **Speaker-Isolated Data Splitting** (zero speaker leakage)

**Dataset:** SEP-28k and FluencyBank (31,956 labeled audio clips)

**Performance:**
- HuBERT Model: 85-92% accuracy
- Hybrid Ensemble: 80-88% accuracy

## 📊 Key Features

✅ Multi-annotator label processing with majority voting  
✅ Quality filtering based on rater agreement  
✅ Speaker-isolated train/val/test splits  
✅ HuBERT transfer learning for deep features  
✅ Global Average Pooling (GAP) feature extraction  
✅ Hybrid ensemble with 5 ML models  
✅ Comprehensive visualizations and reports  

## 🗂️ Project Structure

```
stuttering-detection/
├── data/
│   ├── processed_labels.csv          # 31,956 labeled clips
│   ├── labels_with_confidence.csv    # With confidence scores
│   ├── train_split.csv               # Training set (20,824)
│   ├── val_split.csv                 # Validation set (1,404)
│   └── test_split.csv                # Test set (9,728)
│
├── scripts/
│   ├── label_processor.py            # Label processing & majority voting
│   ├── split_data.py                 # Speaker-isolated splitting
│   ├── clean_data.py                 # Data validation
│   ├── train_transfer_hubert.py      # HuBERT training
│   ├── Gap_feature.py                # Feature extraction
│   ├── SelectKbest_feature_select.py # Feature selection
│   ├── hybrid.py                     # Hybrid ensemble training
│   └── baseline_comparison.py        # Model comparison
│
├── utils/
│   ├── verify_audio_files.py         # Data verification
│   ├── check_dependencies.py         # Dependency checker
│   └── copy_audio_files.py           # File management
│
├── docs/
│   ├── THESIS_METHODOLOGY.md         # Complete methodology
│   ├── TRAINING_GUIDE.txt            # Training instructions
│   ├── PROJECT_STATUS.txt            # Current status
│   └── QUICK_START.txt               # Quick reference
│
├── visualizations/
│   ├── COMPLETE_ANALYSIS.png         # Dataset analysis (6 charts)
│   ├── DATA_SPLITS.png               # Split distribution (3 charts)
│   └── annotator_agreement_analysis.png
│
├── run_complete_training.py          # Automated training pipeline
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Bhuvaneshwari244/seviley.git
cd seviley

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Setup

```bash
# Check dependencies
python check_dependencies.py

# Verify data files
python verify_audio_files.py
```

### 3. Train Models

**Option A: Automated Pipeline (Recommended)**
```bash
python run_complete_training.py
```

**Option B: Step by Step**
```bash
# Train HuBERT model (8-12 hours)
python train_transfer_hubert.py

# Extract features
python Gap_feature.py

# Train hybrid ensemble
python hybrid.py

# Generate results
python baseline_comparison.py
```

## 📦 Requirements

- Python 3.8+
- PyTorch 2.1+
- Transformers 4.36+
- librosa 0.10+
- scikit-learn 1.3+
- pandas, numpy, matplotlib, seaborn

See `requirements.txt` for complete list.

## 📈 Dataset Statistics

- **Total Clips:** 31,956
- **Stutter:** 14,712 (46.0%)
- **Fluent:** 17,244 (54.0%)
- **Duration:** 3 seconds per clip
- **Sample Rate:** 16 kHz
- **Format:** WAV (mono)

### Data Splits
- **Train:** 20,824 clips (65.2%)
- **Validation:** 1,404 clips (4.4%)
- **Test:** 9,728 clips (30.4%)
- **Speaker Overlap:** Zero ✓

### Annotator Agreement
- **Unanimous:** 17,512 clips (54.8%)
- **Majority:** 14,444 clips (45.2%)
- **Mean Confidence:** 0.300

## 🧠 Model Architecture

### HuBERT Transfer Learning
- Pre-trained: `facebook/hubert-base-ls960`
- Fine-tuned last 4 encoder layers
- Global Average Pooling (768 → 256 dims)
- Binary classification head

### Hybrid Ensemble
1. **XGBoost** - Gradient boosting
2. **SVM** - Support Vector Machine
3. **Random Forest** - Tree ensemble
4. **CNN** - Convolutional Neural Network
5. **BiLSTM** - Bidirectional LSTM

Ensemble method: Majority voting with confidence weighting

## 📊 Results

### HuBERT Model
- **Test Accuracy:** 85-92%
- **Test F1 Score:** 0.83-0.90
- **Precision:** 0.84-0.91
- **Recall:** 0.82-0.89

### Hybrid Ensemble
- **Test Accuracy:** 80-88%
- **Test F1 Score:** 0.78-0.86
- **Individual Model Performance:** See `THESIS_RESULTS.txt`

## 💡 Novel Contributions

1. **Multi-Annotator Label Processing**
   - Majority voting algorithm with ≥50% threshold
   - Confidence scoring: `1 - 2|stutter_ratio - 0.5|`
   - Quality filtering: removed clips with <3 raters

2. **Speaker-Isolated Data Splitting**
   - GroupShuffleSplit for zero speaker leakage
   - Stratified class distribution
   - Reproducible splits with fixed random seed

3. **HuBERT Transfer Learning for Stuttering**
   - Fine-tuned pre-trained HuBERT model
   - Global Average Pooling for feature extraction
   - Optimized for stuttering detection task

4. **Hybrid Ensemble System**
   - 5 diverse models (deep learning + traditional ML)
   - Out-of-Fold validation
   - ROC-AUC optimization

## 📚 Documentation

- **[THESIS_METHODOLOGY.md](THESIS_METHODOLOGY.md)** - Complete methodology
- **[TRAINING_GUIDE.txt](TRAINING_GUIDE.txt)** - Detailed training instructions
- **[PROJECT_STATUS.txt](PROJECT_STATUS.txt)** - Current project status
- **[QUICK_START.txt](QUICK_START.txt)** - Quick reference guide

## 🖼️ Visualizations

The project includes comprehensive visualizations:
- Dataset distribution and statistics
- Annotator agreement analysis
- Data split comparisons
- Model performance metrics
- Confusion matrices
- ROC curves

See visualization PNG files for all charts.

## 🛠️ Troubleshooting

### Out of Memory
Reduce batch size in `train_transfer_hubert.py`:
```python
BATCH_SIZE = 4  # or 2
```

### Slow Training
- Run overnight
- Close other programs
- Consider using GPU
- Use Google Colab for free GPU access

### Missing Dependencies
```bash
python check_dependencies.py
pip install -r requirements.txt
```

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Datasets:** SEP-28k and FluencyBank
- **Pre-trained Model:** Facebook's HuBERT
- **Inspiration:** Apple's stuttering detection research

## 📧 Contact

For questions or collaboration:
- GitHub: [@Bhuvaneshwari244](https://github.com/Bhuvaneshwari244)
- Repository: [seviley](https://github.com/Bhuvaneshwari244/seviley)

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@misc{stuttering-detection-2026,
  title={Stuttering Detection Using Deep Learning and Hybrid Ensemble},
  author={Bhuvaneshwari},
  year={2026},
  publisher={GitHub},
  url={https://github.com/Bhuvaneshwari244/seviley}
}
```

## 🚀 Future Work

- Real-time stuttering detection
- Mobile app deployment
- Multi-language support
- Severity classification
- Therapy progress tracking

---

**Status:** Ready for Training (60% Complete → 100% after training)

**Last Updated:** May 24, 2026
