# Preprocessing Report

## Image Resizing

All chest X-ray images were resized to 224×224 pixels.

## Channel Configuration

Original grayscale images were converted to 3-channel images for compatibility with CNN architectures and future transfer learning experiments.

## Normalization

Pixel values were normalized using:

Mean = [0.5, 0.5, 0.5]

Std = [0.5, 0.5, 0.5]

Resulting pixel range approximately [-1, 1].

## Data Augmentation

Training images were augmented using:

- Random Horizontal Flip
- Random Rotation (±10°)
- Random Translation (5%)

Validation and test images were not augmented.

## Stratified Split

The original training dataset was split into:

- Training Set: 80%
- Validation Set: 20%

using stratified sampling.

The original test set remained unchanged.

## Final Tensor Shape

(3,224,224)

## Conclusion

The preprocessing pipeline successfully converted raw chest X-ray images into CNN-ready tensors suitable for centralized and federated learning experiments.
