# Neural Network Model Reverse Engineering Suite

A comprehensive toolkit for analyzing and reverse engineering neural network models from various frameworks. This suite consists of two main scripts: a model downloader and a detailed analyzer that extracts architectural details, performance characteristics, and deployment insights from black-box models.

## Motivating Works
Satwik Kundu and Swaroop Ghosh. 2024. SoK Paper: Security Concerns in Quantum Machine Learning as a Service. In Proceedings of the International Workshop on Hardware and Architectural Support for Security and Privacy 2024 (HASP '24). Association for Computing Machinery, New York, NY, USA, 28–36. https://doi.org/10.1145/3696843.3696846

Meng Shi, Wei Lin, and Wenbo Deng. 2024. Research on Key Techniques for Reverse Engineering of Deep Learning Models for x86 Executable Files. In Proceedings of the 2024 7th International Conference on Computer Information Science and Artificial Intelligence (CISAI '24). Association for Computing Machinery, New York, NY, USA, 148–153. https://doi.org/10.1145/3703187.3703212

## Overview

This toolkit enables AI developers, researchers, and engineers to:

- **Reverse engineer** neural network architectures from model files
- **Extract detailed specifications** including parameters, layers, and operations
- **Analyze performance characteristics** and hardware requirements
- **Detect dataset types** and intended use cases
- **Assess security vulnerabilities** and deployment readiness
- **Download diverse models** from Hugging Face for testing and analysis

## Scripts Included

### 1. Model Downloader (`model_downloader.py`)
Downloads various types of neural network models from Hugging Face and creates sample models for analysis testing.

### 2. Model Analyzer (`model_analyzer.py`)
Performs comprehensive reverse engineering analysis on neural network model files, extracting architectural details and providing deployment insights.

## Supported Model Formats

### Framework Support
- **ONNX** (.onnx) - Cross-platform neural network format
- **TensorFlow/Keras** (.h5, .hdf5) - TensorFlow SavedModel format
- **PyTorch** (.pt, .pth, .bin) - PyTorch model files and state dictionaries
- **Scikit-learn** (.pkl, .joblib) - Traditional ML models

### Model Types Analyzed
- Convolutional Neural Networks (CNNs)
- Transformer models (BERT, GPT, etc.)
- Recurrent Neural Networks (RNNs, LSTMs)
- Dense/Fully Connected Networks
- Ensemble models (Random Forest, etc.)
- Support Vector Machines
- Linear/Logistic Regression models

## Analysis Capabilities

### Architecture Analysis
- **Layer-by-layer breakdown** with parameter counts
- **Operation type distribution** and computational graph analysis
- **Model topology** including skip connections and branching
- **Parameter distribution** across layers and modules
- **Weight tensor analysis** with shapes and data types

### Performance Profiling
- **FLOPS estimation** for computational complexity
- **Inference speed benchmarking** with real-time measurements
- **Memory usage analysis** including RAM and GPU requirements
- **Throughput calculations** (frames per second, requests per second)
- **Model size optimization** recommendations

### Dataset Detection
- **Automatic dataset type identification** based on input dimensions
- **Confidence scoring** for dataset predictions
- **Input preprocessing requirements** and normalization parameters
- **Expected data formats** and validation requirements
- **Business use case suggestions** based on model characteristics

### Security Assessment
- **Vulnerability analysis** including adversarial attack susceptibility
- **Robustness evaluation** and failure mode identification
- **Input validation requirements** and safety recommendations
- **Privacy implications** and data leakage risks
- **Compliance considerations** (GDPR, bias assessment)

### Deployment Intelligence
- **Hardware requirement estimation** (CPU, GPU, RAM)
- **Platform compatibility matrix** (mobile, web, edge, cloud)
- **Optimization recommendations** (quantization, pruning)
- **API deployment options** and containerization readiness
- **Monitoring and logging suggestions** for production environments

### Business Context
- **Use case identification** and application domains
- **Performance expectations** and accuracy benchmarks
- **Licensing information** and usage restrictions
- **ROI considerations** and maintenance requirements

## Installation

### Prerequisites
```bash
# Install in Google Colab
!pip install huggingface_hub requests onnx psutil

# For TensorFlow models
!pip install tensorflow

# For PyTorch models  
!pip install torch

# For scikit-learn models
!pip install scikit-learn
```

### Quick Start
1. Upload both scripts to your Google Colab environment
2. Run the model downloader to get sample models
3. Run the analyzer on any model file
4. Review the comprehensive analysis output

## Usage

### Downloading Models
```python
# Run the downloader script
python model_downloader.py

# Select from options:
# 1. Download ALL models
# 2. Download by category (vision, NLP, etc.)
# 3. Download specific models
# 4. Quick start pack (small models for testing)
```

### Analyzing Models
```python
# Run the analyzer script
python model_analyzer.py

# The script will:
# 1. Scan /content/ directory for models
# 2. Display found models with sizes
# 3. Allow selection of model to analyze
# 4. Perform comprehensive analysis
# 5. Display results in formatted tables
```

## Example Output

```
================================================================================
COMPREHENSIVE AI MODEL ANALYSIS
================================================================================
File Name:           mnist_digit_classifier.onnx
File Size:           26.85 KB
File Extension:      .onnx
Memory Footprint:    32.22 MB (estimated in RAM)
--------------------------------------------------------------------------------
ONNX MODEL ANALYSIS
--------------------------------------------------------------------------------
Producer:            pytorch
Producer Version:    1.9
Total Parameters:    21,840
Weight Tensors:      8
Parameters Memory:   85.31 KB
Model Overhead:      15.54 KB
Total Operations:    8
Compute-heavy Ops:   2

Operation Breakdown:
  Conv                   2 ( 25.0%)
  MaxPool                2 ( 25.0%)
  Relu                   2 ( 25.0%)
  Flatten                1 ( 12.5%)
  Gemm                   1 ( 12.5%)

DATASET & USE CASE ANALYSIS
--------------------------------------------------------------------------------
Dataset Type:        MNIST Handwritten Digits
Confidence:          High
Task:                Digit Recognition
Business Use:        Document digitization, form processing
Data Source:         28x28 grayscale images
Accuracy Expectation: 99%+
Licensing:           Public domain

HARDWARE REQUIREMENTS
--------------------------------------------------------------------------------
Minimum Ram:         4 GB
Recommended Ram:     8 GB
Gpu Memory:          2 GB
Cpu Cores:           2
Storage:             32 MB
```

## Use Cases

### Research Applications
- **Model archaeology** - Understanding architectures of published models
- **Benchmark comparison** - Analyzing competing model architectures
- **Transfer learning** - Identifying suitable base models
- **Academic analysis** - Studying neural network design patterns

### Development Applications  
- **Model validation** - Verifying model specifications before deployment
- **Performance optimization** - Identifying computational bottlenecks
- **Architecture debugging** - Understanding unexpected model behavior
- **Deployment planning** - Estimating infrastructure requirements

### Security Applications
- **Model auditing** - Assessing models for vulnerabilities
- **Compliance checking** - Ensuring models meet regulatory requirements  
- **Intellectual property** - Analyzing model architectures for patents
- **Risk assessment** - Evaluating models before production deployment

### Business Applications
- **Due diligence** - Evaluating acquired or licensed models
- **Competitive analysis** - Understanding competitor model capabilities
- **Resource planning** - Estimating deployment costs and requirements
- **Technology assessment** - Evaluating AI solutions before adoption

## Model Categories Supported

### Computer Vision Models
- Image classification (ResNet, VGG, MobileNet)
- Object detection (YOLO, R-CNN variants)
- Semantic segmentation models
- Generative models (GANs, VAEs)

### Natural Language Processing Models
- Transformer models (BERT, GPT, T5)
- Recurrent models (LSTM, GRU)
- Sequence-to-sequence models
- Language generation models

### Traditional Machine Learning Models
- Tree-based models (Random Forest, Gradient Boosting)
- Linear models (Logistic Regression, SVM)
- Clustering algorithms (K-means, DBSCAN)
- Dimensionality reduction models (PCA, t-SNE)

### Audio and Multimodal Models
- Speech recognition models (Wav2Vec, Whisper)
- Audio classification models
- Multimodal vision-language models
- Time series prediction models

## Technical Specifications

### Analysis Depth
- **Parameter counting** with granular breakdown by layer/operation
- **Memory footprint calculation** including overhead and optimization potential
- **Computational complexity estimation** using FLOPS and operation analysis
- **Data flow analysis** through the neural network graph
- **Quantization readiness assessment** for model optimization

### Performance Metrics
- **Inference latency** measured through actual benchmarks
- **Throughput estimation** for batch and streaming scenarios
- **Memory bandwidth requirements** for optimal performance
- **Hardware utilization efficiency** recommendations

### Security Analysis
- **Attack surface evaluation** for adversarial vulnerabilities
- **Input validation requirements** and sanitization needs
- **Model fingerprinting** for intellectual property protection
- **Privacy risk assessment** including potential data leakage

## Limitations

- Analysis quality depends on model file completeness
- Performance benchmarks are estimates and may vary on different hardware
- Security assessment provides general recommendations, not exhaustive audits
- Dataset detection relies on heuristics and may require manual verification
- Some proprietary model formats may not be fully supported

## Contributing

This toolkit is designed for educational and research purposes. Contributions welcome for:
- Additional model format support
- Enhanced security analysis capabilities  
- Improved performance benchmarking
- Better dataset detection algorithms
- Extended deployment platform support

## License

This project is provided for educational and research purposes. Users are responsible for ensuring compliance with model licenses and applicable regulations when analyzing third-party models.
