# Deep-Learning-for-Computer-Vision

Topics Covered:

Convolutional Neural Network (CNN)
  - Intro to CNN
  - Problems that CNN's solve
  - Applications of CNN
  - Convolution Operation
  - Padding and Strides
  - Pooling
  - CNN vs ANN
  - Final Intuition
  - Back Propagation on CNN (Mathematically deriving)
  - Data Augmentation
  - Pre-Trained Models in CNN
    - LeNET
    - AlexNET
    - VGG16
    - GoogleNET (Inception)
    - ReSNET
  - Transfer Learning
      - Feature Extration Transfer Learning
      - Fine Tuning Transfer Learning

- Object Detection
  - Semantic and Instance Segmentation
  - GAN's
  - Unet Architecture
  - RCNN, Fast RCNN and Faster RCNN
  - Yolo Architecture

- Final Self Driving car project



# **Also, Here are all the coding files that were done while understanding the topic**

Below is my explaination on the CNN architecture. I have used claude in order to rewrite and make readme look better :) But the words and intuition is written by me
# Understanding CNN Architecture
### A Complete Visual Walkthrough

<div align="center">

*A comprehensive guide to understanding how Convolutional Neural Networks process images*

[Overview](#overview) • [Architecture](#architecture-breakdown) • [Process](#step-by-step-process) • [Key Insights](#key-takeaways)

</div>

---

## Overview

This guide walks through how a CNN processes a single image (224×224×3) from input to output, breaking down each component with practical insights and real-world understanding.

**What You'll Learn:**
- How convolutional layers extract features
- The relationship between filter dimensions and input depth
- How pooling reduces spatial dimensions
- The complete flow from pixels to predictions

---

## Architecture Breakdown

### Input Specifications
```
Image Dimensions: 224 × 224 × 3
    ├── Height: 224 pixels
    ├── Width: 224 pixels
    └── Channels: 3 (RGB)
```

### Network Flow
```
Input (224×224×3)
    ↓
Conv Layer 1 (64 filters, 3×3×3)
    ↓
ReLU Activation
    ↓
Max Pooling (2×2)
    ↓
Conv Layer 2 (128 filters, 3×3×64)
    ↓
ReLU Activation
    ↓
Max Pooling (2×2)
    ↓
Flatten
    ↓
Fully Connected Layers
    ↓
Output (Softmax/Sigmoid)
```

---

## Step-by-Step Process

### Step 1: Input Image Representation

A colored input image is provided. The image is already in pixel values — that's how machines read images. Since it's a colored image, it has 3 channels for RGB.

---

### Step 2: Filter Initialization

We have randomly initialized weights for our filters/kernels. Let's say we start with **64 filters**. Each filter will learn to detect different patterns during training.

**Important:** We don't manually assign what each filter should look for, but we do decide:
- How many filters we want
- What their dimensions should be

---

### Step 3: Forward Propagation Begins

The input starts moving through the network. This is where the feature extraction begins.

---

### Step 4: Filter Depth Must Match Input Depth

> **Critical Rule:** Filter depth MUST equal input depth

Since our input has 3 channels (RGB), our filters must also have 3 channels. So if we're using 3×3 filters, they're actually **3×3×3** (height × width × depth).
```
Filter Dimensions:
├── Spatial: 3×3 (height × width)
└── Depth: 3 (must match input channels)
```

---

### Step 5: The Convolution Operation

**How it works:**

The filter slides across the input image step by step — moving through each row until it reaches the end, then shifting down and repeating. Think of it like a small cuboid (the filter) sliding through subsets of a larger cuboid (the image).

**The Parallel Processing Insight:**

We have 64 filters, and they all process the image **in parallel**. Each filter:

1. Takes a 3×3×3 chunk of the input
2. Does element-wise multiplication with its weights (all 27 values)
3. Sums everything up into a single scalar value
4. This scalar becomes one pixel in that filter's feature map
```python
# Conceptual representation
for each position in image:
    for each of 64 filters (in parallel):
        output_value = sum(input_chunk * filter_weights)
        feature_map[filter_id][position] = output_value
```

**Result:** In one sliding step, all 64 filters process the same image patch simultaneously. Each produces one value. This means all 64 feature maps get their first pixel filled at once.

---

### Step 6: Activation Function

Now we apply **ReLU** to these feature maps.
```
ReLU(x) = max(0, x)
```

Simple rule: any negative values become 0. This adds non-linearity to our network.

---

### Step 7: Padding (Optional but Important)

We can apply padding to the input before or during convolution.

**Why padding matters:**

| Without Padding | With Padding |
|----------------|--------------|
| Edge pixels used in fewer operations | Edge pixels participate more |
| Network favors center features | Balanced feature importance |
| Spatial dimensions shrink | Can maintain dimensions |

**The insight:** Without padding, pixels on the edges and corners get used in fewer convolution operations compared to center pixels. This means the network indirectly gives more importance to inner features. By using padding (adding zeros around the edges), we let edge values participate in more operations, giving them more importance too.

---

### Step 8: Pooling Layer

Each of the 64 feature maps now goes through a pooling function. Let's use **max pooling** with a 2×2 window:
```
Input:  224×224×64
Output: 112×112×64
```

**What pooling does:**
- Reduces spatial dimensions (height and width)
- Keeps depth (number of channels) the same
- Downsamples while preserving the most important features

---

### Step 9: Second Convolution Layer

Now the process repeats. But here's the crucial part:

> **Key Rule:** New filter depth MUST match previous layer's output depth

Let's say we want 128 filters this time, each with size 3×3. The third dimension (depth) **must match the previous layer's output depth**. So our filters are now **3×3×64**, not 3×3×3.
```
Previous Layer Output: 112×112×64
Filter Dimensions: 3×3×64
Number of Filters: 128
Next Layer Output: 112×112×128 (with padding)
```

Each of these 128 filters looks at **all 64 channels** from the previous layer simultaneously and produces one feature map.

**Complete flow for this layer:**
```
Input:  112×112×64  
    ↓ Convolution (128 filters, 3×3×64, with padding)
Output: 112×112×128
    ↓ ReLU
Output: 112×112×128
    ↓ Max Pooling (2×2)
Output: 56×56×128
```

---

### Step 10: Deeper Layers

This same process repeats through multiple layers.

**Common pattern observed:**

| Layer | Spatial Dimensions | Number of Channels |
|-------|-------------------|--------------------|
| Conv1 | 224×224 | 64 |
| Pool1 | 112×112 | 64 |
| Conv2 | 112×112 | 128 |
| Pool2 | 56×56 | 128 |
| Conv3 | 56×56 | 256 |
| Pool3 | 28×28 | 256 |
| Conv4 | 28×28 | 512 |
| Pool4 | 14×14 | 512 |

**Pattern:**
- Spatial dimensions decrease: 224 → 112 → 56 → 28 → 14...
- Number of filters increases: 64 → 128 → 256 → 512...

**Why this works:**
- Early layers learn simple features (edges, colors)
- Deeper layers learn complex features (shapes, objects, textures)
- The network figures this out automatically during training through backpropagation

---

### Step 11: Flattening

Eventually, we reach our final convolutional layer. Let's say we end up with a tensor of size **7×7×512**.

We flatten this into a 1D array:
```
7 × 7 × 512 = 25,088 neurons
```

This 1D array becomes the input to our fully connected layers.
```python
# Transformation
Input:  [7, 7, 512]  # 3D tensor
Output: [25088]      # 1D vector
```

---

### Step 12: Fully Connected Layers

These flattened values now pass through traditional neural network layers (dense/fully connected layers).

**What we apply:**
- Different activation functions (ReLU for hidden layers)
- Different optimizers (Adam, SGD, etc.)
- Dropout for regularization (optional)
```
Example Architecture:
25,088 neurons
    ↓ Dense Layer
4,096 neurons + ReLU + Dropout
    ↓ Dense Layer
4,096 neurons + ReLU + Dropout
    ↓ Dense Layer
1,000 neurons (output classes)
```

---

### Step 13: Output Layer

At the final layer, we get our predictions:

**Binary Classification:**
```python
Activation: Sigmoid
Output: Single probability [0, 1]
Interpretation: P(class = positive)
```

**Multi-class Classification:**
```python
Activation: Softmax
Output: Probability distribution across all classes
Example: [0.05, 0.82, 0.03, 0.10] for 4 classes
Interpretation: Class 2 has 82% probability
```

---

## Key Takeaways

<div align="center">

### Core Principles

</div>

| Principle | Explanation |
|-----------|-------------|
| **Filter Depth Matching** | Filter depth ALWAYS matches previous layer's output depth |
| **Parallel Processing** | All filters in a layer process simultaneously, creating feature maps in parallel |
| **Architecture Design** | You choose layers, filters, sizes — network learns actual filter values |
| **Dimension Pattern** | Spatial dimensions decrease, depth increases as you go deeper |
| **Padding Purpose** | Preserves spatial information and balances edge/center feature importance |
| **Pooling Function** | Reduces dimensions while keeping the most important features |

---

### The Learning Process
```
┌─────────────────────────────────────────────────────────────┐
│  YOU DESIGN                │  NETWORK LEARNS                 │
├────────────────────────────┼─────────────────────────────────┤
│  • Number of layers        │  • Filter weight values         │
│  • Filters per layer       │  • What patterns to detect      │
│  • Filter sizes            │  • Feature hierarchies          │
│  • Activation functions    │  • Optimal representations      │
│  • Optimizer choice        │                                 │
└────────────────────────────┴─────────────────────────────────┘
```

---

### Dimension Tracking Example
```
Input Image
224×224×3
    ↓
Conv1: 64 filters (3×3×3) + ReLU + Padding
224×224×64
    ↓
MaxPool (2×2, stride=2)
112×112×64
    ↓
Conv2: 128 filters (3×3×64) + ReLU + Padding
112×112×128
    ↓
MaxPool (2×2, stride=2)
56×56×128
    ↓
Conv3: 256 filters (3×3×128) + ReLU + Padding
56×56×256
    ↓
MaxPool (2×2, stride=2)
28×28×256
    ↓
Conv4: 512 filters (3×3×256) + ReLU + Padding
28×28×512
    ↓
MaxPool (2×2, stride=2)
14×14×512
    ↓
Flatten
100,352 neurons
    ↓
Fully Connected + Softmax
1,000 classes
```

---

## Visual Summary

### The Big Picture
```
┌──────────────────────────────────────────────────────────────────┐
│                    CNN FEATURE EXTRACTION                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Early Layers (64-128 filters)                                   │
│  ├── Large spatial dimensions (224×224 → 112×112)                │
│  ├── Few channels                                                 │
│  └── Learn: edges, colors, simple textures                       │
│                                                                   │
│  Middle Layers (128-256 filters)                                 │
│  ├── Medium spatial dimensions (56×56 → 28×28)                   │
│  ├── More channels                                                │
│  └── Learn: shapes, patterns, object parts                       │
│                                                                   │
│  Deep Layers (256-512 filters)                                   │
│  ├── Small spatial dimensions (14×14 → 7×7)                      │
│  ├── Many channels                                                │
│  └── Learn: complex objects, full features                       │
│                                                                   │
│  Classification Layers                                            │
│  ├── Flattened representation                                     │
│  ├── Fully connected                                              │
│  └── Output: class probabilities                                 │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Notes

> Some calculations in this explanation are simplified for clarity. In practice, exact dimensions depend on stride, padding type, and filter sizes used.

---

<div align="center">

**Built with understanding** | **Refined for clarity** | **Shared for learning**

[Back to Top](#understanding-cnn-architecture)

</div>
