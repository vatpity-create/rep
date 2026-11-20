# Individual project proposal

## Lightweight CNN for Clothing Image Classification

Student: Ihor Velychko

Student ID: Not provided

Approval: Pending instructor review

### Problem

The project aims to develop a lightweight convolutional neural network (CNN) that classifies clothing images on devices with limited computational resources. The main challenge is to maintain good classification accuracy while reducing model size and inference time.

### Dataset

Fashion-MNIST contains 70,000 grayscale images (28 × 28 pixels) from 10 clothing categories. The official 60,000 training images will be split into 54,000 training and 6,000 validation samples using a fixed seed. The remaining 10,000 official test images will be reserved for final evaluation.

### Success metrics

The optimized model should achieve at least 90% test accuracy, with a target accuracy loss of no more than 1 percentage point relative to the unoptimized baseline. The deployment metrics are a model size of ≤ 5 MB and inference latency of ≤ 50 ms.

### Deployment constraints

The model will run on a standard CPU using a batch size of one, without a dedicated GPU. After a warm-up, latency will be measured over 100 predictions and reported as median and 95th-percentile inference time; the latency target applies to the 95th percentile. Model size will be measured from the serialized deployment artifact.

### Planned optimization

Train a compact CNN in PyTorch as a baseline, then apply INT8 post-training static quantization using representative training samples for calibration. Explore structured channel pruning followed by fine-tuning as a secondary approach. Compare baseline, quantized and, if feasible, pruned models using the same test set and CPU benchmark.

### Expected results

A reproducible training and evaluation pipeline and a compact, CPU-ready classification model. The final results will compare accuracy, inference latency and model size before and after optimization, and indicate whether all deployment constraints have been met.
