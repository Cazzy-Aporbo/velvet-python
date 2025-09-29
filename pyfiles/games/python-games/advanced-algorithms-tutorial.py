"""
Advanced Machine Learning Algorithms: Comprehensive Implementation and Tutorial
Author: Cazzy Aporbo 2025
Python 3.8+
Dependencies: numpy, scipy, sklearn, torch (optional for deep learning sections)

This file provides production-ready implementations of cutting-edge ML algorithms
with detailed explanations of when, why, and how to use each one.
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import math
from collections import defaultdict, deque
from scipy import stats, linalg
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import time
import random


class CNNFeatureExtractor:
    """
    Convolutional Neural Network for spatial representation learning.
    
    Use when: You have grid-like data (images, spectrograms) with local patterns
    Similar to: Vision Transformers (ViT), Graph Convolutions
    
    Key insight: CNNs exploit spatial locality through weight sharing in convolutions,
    making them parameter-efficient for vision tasks.
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], num_filters: int = 32):
        """
        Initialize CNN architecture.
        
        Args:
            input_shape: (height, width, channels) of input
            num_filters: Number of convolutional filters (feature detectors)
        """
        self.input_shape = input_shape
        self.num_filters = num_filters
        # In production, these would be learned parameters
        self.conv_weights = self._initialize_conv_weights()
        self.pooling_size = 2
        
    def _initialize_conv_weights(self) -> np.ndarray:
        """
        Initialize convolutional kernels using Xavier/He initialization.
        
        Xavier init: Good for tanh/sigmoid activations
        He init: Better for ReLU activations (what we use here)
        
        Returns:
            4D array of shape (num_filters, kernel_h, kernel_w, input_channels)
        """
        kernel_size = 3  # 3x3 is standard for modern CNNs
        _, _, input_channels = self.input_shape
        
        # He initialization: std = sqrt(2 / fan_in)
        # fan_in = kernel_size * kernel_size * input_channels
        fan_in = kernel_size * kernel_size * input_channels
        std_dev = np.sqrt(2.0 / fan_in)
        
        weights = np.random.normal(0, std_dev, 
                                  (self.num_filters, kernel_size, kernel_size, input_channels))
        
        return weights
    
    def convolve2d(self, image: np.ndarray, kernel: np.ndarray, 
                   stride: int = 1, padding: str = 'same') -> np.ndarray:
        """
        Perform 2D convolution operation.
        
        This is the core operation of CNNs. Each filter slides across the image,
        computing dot products to detect specific patterns.
        
        Args:
            image: Input image (H, W, C)
            kernel: Convolution kernel (kH, kW, C)
            stride: Step size for sliding window
            padding: 'same' maintains dimensions, 'valid' reduces them
            
        Returns:
            Feature map after convolution
        """
        h, w, _ = image.shape
        kh, kw, _ = kernel.shape
        
        # Calculate padding needed for 'same' convolution
        if padding == 'same':
            pad_h = (kh - 1) // 2
            pad_w = (kw - 1) // 2
            # Pad image with zeros (most common padding strategy)
            image_padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), 
                                mode='constant', constant_values=0)
        else:
            image_padded = image
            
        # Calculate output dimensions
        out_h = (image_padded.shape[0] - kh) // stride + 1
        out_w = (image_padded.shape[1] - kw) // stride + 1
        
        # Initialize output feature map
        output = np.zeros((out_h, out_w))
        
        # Sliding window convolution
        for i in range(0, out_h):
            for j in range(0, out_w):
                # Extract receptive field (local region)
                h_start = i * stride
                h_end = h_start + kh
                w_start = j * stride
                w_end = w_start + kw
                
                receptive_field = image_padded[h_start:h_end, w_start:w_end, :]
                
                # Compute convolution as element-wise multiply and sum
                # This detects how much the kernel pattern matches this region
                output[i, j] = np.sum(receptive_field * kernel)
                
        return output
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """
        ReLU activation: f(x) = max(0, x)
        
        Why ReLU?
        1. Avoids vanishing gradients (unlike sigmoid/tanh)
        2. Sparse activation (biological plausibility)
        3. Computationally efficient
        """
        return np.maximum(0, x)
    
    def max_pool2d(self, feature_map: np.ndarray, pool_size: int = 2) -> np.ndarray:
        """
        Max pooling: Downsamples by taking maximum in each region.
        
        Benefits:
        1. Translation invariance (small shifts don't affect output)
        2. Reduces parameters (prevents overfitting)
        3. Increases receptive field
        
        Args:
            feature_map: Input feature map (H, W)
            pool_size: Size of pooling window
            
        Returns:
            Downsampled feature map
        """
        h, w = feature_map.shape
        out_h = h // pool_size
        out_w = w // pool_size
        
        pooled = np.zeros((out_h, out_w))
        
        for i in range(out_h):
            for j in range(out_w):
                # Take maximum value in pooling window
                # This preserves the strongest activation (most important feature)
                h_start = i * pool_size
                h_end = h_start + pool_size
                w_start = j * pool_size
                w_end = w_start + pool_size
                
                pooled[i, j] = np.max(feature_map[h_start:h_end, w_start:w_end])
                
        return pooled
    
    def forward(self, image: np.ndarray) -> np.ndarray:
        """
        Forward pass through CNN layers.
        
        Architecture: Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool
        This is a simplified LeNet/AlexNet style architecture.
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            Extracted feature vector
        """
        features = []
        
        # First convolutional layer
        for i in range(min(self.num_filters, 5)):  # Limit for demo
            # Each filter detects different patterns (edges, textures, etc.)
            kernel = self.conv_weights[i]
            
            # Convolution: Pattern detection
            conv_output = self.convolve2d(image, kernel)
            
            # ReLU: Non-linearity (allows learning complex functions)
            activated = self.relu(conv_output)
            
            # Pooling: Spatial downsampling
            pooled = self.max_pool2d(activated, self.pooling_size)
            
            features.append(pooled)
            
        # Stack all feature maps
        feature_maps = np.stack(features, axis=-1)
        
        # Global average pooling (modern technique, better than flattening)
        # Reduces each feature map to single value
        global_features = np.mean(feature_maps, axis=(0, 1))
        
        return global_features


class LSTMSequenceModel:
    """
    Long Short-Term Memory for sequence modeling with memory.
    
    Use when: Sequential data with long-term dependencies (text, time series, speech)
    Similar to: Transformers, Temporal CNNs, GRUs
    
    Key insight: LSTMs solve vanishing gradient problem through gating mechanisms
    that control information flow.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize LSTM cell.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden state (memory capacity)
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weight matrices
        # LSTM has 4 gates: input, forget, output, candidate
        self.weights = self._initialize_weights()
        
    def _initialize_weights(self) -> Dict[str, np.ndarray]:
        """
        Initialize LSTM weights using proper initialization.
        
        Each gate needs weights for input and hidden state.
        Glorot uniform initialization is standard for LSTMs.
        """
        weights = {}
        
        # Scale for Glorot uniform initialization
        input_scale = np.sqrt(6.0 / (self.input_dim + self.hidden_dim))
        hidden_scale = np.sqrt(6.0 / (self.hidden_dim + self.hidden_dim))
        
        # For each gate (input, forget, output, candidate)
        for gate in ['i', 'f', 'o', 'g']:
            # Weights for input
            weights[f'W_{gate}'] = np.random.uniform(
                -input_scale, input_scale, 
                (self.input_dim, self.hidden_dim)
            )
            # Weights for hidden state
            weights[f'U_{gate}'] = np.random.uniform(
                -hidden_scale, hidden_scale,
                (self.hidden_dim, self.hidden_dim)
            )
            # Bias terms
            weights[f'b_{gate}'] = np.zeros(self.hidden_dim)
            
        # Initialize forget gate bias to 1.0 (important trick)
        # This helps LSTM remember by default rather than forget
        weights['b_f'] += 1.0
        
        return weights
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation for gates (outputs between 0 and 1)."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
    
    def tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh activation for candidate values (outputs between -1 and 1)."""
        return np.tanh(x)
    
    def forward_step(self, x_t: np.ndarray, h_prev: np.ndarray, 
                     c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single LSTM forward step.
        
        The LSTM equations (the key to understanding LSTMs):
        1. Forget gate: f_t = σ(W_f·x_t + U_f·h_{t-1} + b_f)
        2. Input gate: i_t = σ(W_i·x_t + U_i·h_{t-1} + b_i)
        3. Candidate: g_t = tanh(W_g·x_t + U_g·h_{t-1} + b_g)
        4. Output gate: o_t = σ(W_o·x_t + U_o·h_{t-1} + b_o)
        5. Cell state: c_t = f_t * c_{t-1} + i_t * g_t
        6. Hidden state: h_t = o_t * tanh(c_t)
        
        Args:
            x_t: Input at time t (input_dim,)
            h_prev: Previous hidden state (hidden_dim,)
            c_prev: Previous cell state (hidden_dim,)
            
        Returns:
            (h_t, c_t): New hidden and cell states
        """
        # Forget gate: Decides what to forget from previous cell state
        # Values close to 0 forget, close to 1 remember
        f_t = self.sigmoid(
            np.dot(x_t, self.weights['W_f']) + 
            np.dot(h_prev, self.weights['U_f']) + 
            self.weights['b_f']
        )
        
        # Input gate: Decides what new information to store
        i_t = self.sigmoid(
            np.dot(x_t, self.weights['W_i']) + 
            np.dot(h_prev, self.weights['U_i']) + 
            self.weights['b_i']
        )
        
        # Candidate values: New information that could be added to cell state
        g_t = self.tanh(
            np.dot(x_t, self.weights['W_g']) + 
            np.dot(h_prev, self.weights['U_g']) + 
            self.weights['b_g']
        )
        
        # Output gate: Decides what parts of cell state to output
        o_t = self.sigmoid(
            np.dot(x_t, self.weights['W_o']) + 
            np.dot(h_prev, self.weights['U_o']) + 
            self.weights['b_o']
        )
        
        # Update cell state: Forget old info and add new info
        # This is the key equation that allows long-term memory
        c_t = f_t * c_prev + i_t * g_t
        
        # Compute hidden state: Filtered version of cell state
        h_t = o_t * self.tanh(c_t)
        
        return h_t, c_t
    
    def forward(self, sequence: np.ndarray) -> np.ndarray:
        """
        Process entire sequence through LSTM.
        
        Args:
            sequence: Input sequence (seq_len, input_dim)
            
        Returns:
            Hidden states for all timesteps (seq_len, hidden_dim)
        """
        seq_len = sequence.shape[0]
        
        # Initialize states
        h_t = np.zeros(self.hidden_dim)
        c_t = np.zeros(self.hidden_dim)
        
        outputs = []
        
        for t in range(seq_len):
            # Process each timestep
            h_t, c_t = self.forward_step(sequence[t], h_t, c_t)
            outputs.append(h_t)
            
        return np.array(outputs)


class SelfAttentionTransformer:
    """
    Transformer with self-attention mechanism.
    
    Use when: Long sequences, parallel processing needed, global dependencies
    Similar to: RNNs, but with parallel processing and better long-range modeling
    
    Key insight: Attention allows direct connections between any positions,
    avoiding sequential bottleneck of RNNs.
    """
    
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        """
        Initialize multi-head self-attention.
        
        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads (parallel attention mechanisms)
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Initialize projection matrices
        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_k = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_v = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_o = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        
    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, 
                                    V: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Core attention mechanism: Attention(Q,K,V) = softmax(QK^T/√d_k)V
        
        This computes how much each position should attend to every other position.
        
        Args:
            Q: Query matrix (seq_len, d_k) - what information to look for
            K: Key matrix (seq_len, d_k) - what information is available
            V: Value matrix (seq_len, d_k) - actual information content
            mask: Optional attention mask for padding or causality
            
        Returns:
            Weighted combination of values based on attention scores
        """
        # Compute attention scores: QK^T
        # Each query attends to all keys
        scores = np.matmul(Q, K.T)
        
        # Scale by sqrt(d_k) to prevent softmax saturation
        # Without scaling, dot products grow with dimension, causing
        # softmax to have extreme values (near 0 or 1)
        scores = scores / np.sqrt(self.d_k)
        
        # Apply mask if provided (e.g., for causal attention or padding)
        if mask is not None:
            scores = scores + mask * -1e9  # Large negative value becomes ~0 after softmax
            
        # Softmax to get attention weights (probabilities)
        attention_weights = self._softmax(scores, axis=-1)
        
        # Apply attention weights to values
        # This creates weighted combination of values based on relevance
        output = np.matmul(attention_weights, V)
        
        return output
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def multi_head_attention(self, x: np.ndarray) -> np.ndarray:
        """
        Multi-head attention: Run attention in parallel with different projections.
        
        Multiple heads allow model to attend to different types of information
        (e.g., syntactic vs semantic in language).
        
        Args:
            x: Input sequence (seq_len, d_model)
            
        Returns:
            Multi-head attention output (seq_len, d_model)
        """
        seq_len = x.shape[0]
        
        # Project to queries, keys, values
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)
        
        # Reshape for multi-head attention
        # Split d_model into num_heads parallel attention computations
        Q = Q.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        
        # Apply attention for each head
        attention_outputs = []
        for head in range(self.num_heads):
            head_output = self.scaled_dot_product_attention(Q[head], K[head], V[head])
            attention_outputs.append(head_output)
            
        # Concatenate heads and project back
        concat_attention = np.concatenate(attention_outputs, axis=-1)
        output = np.matmul(concat_attention, self.W_o)
        
        return output


class VariationalAutoencoder:
    """
    VAE for probabilistic generative modeling with latent variables.
    
    Use when: Need probabilistic latent representation, generation with uncertainty
    Similar to: GANs, Normalizing Flows, standard Autoencoders
    
    Key insight: VAEs learn probabilistic latent space by optimizing ELBO
    (Evidence Lower BOund), enabling both generation and inference.
    """
    
    def __init__(self, input_dim: int, latent_dim: int):
        """
        Initialize VAE architecture.
        
        Args:
            input_dim: Dimension of input data
            latent_dim: Dimension of latent space (bottleneck)
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder network parameters
        self.encoder_weights = {
            'W1': np.random.randn(input_dim, 128) * np.sqrt(2.0 / input_dim),
            'b1': np.zeros(128),
            'W_mu': np.random.randn(128, latent_dim) * np.sqrt(2.0 / 128),
            'b_mu': np.zeros(latent_dim),
            'W_logvar': np.random.randn(128, latent_dim) * np.sqrt(2.0 / 128),
            'b_logvar': np.zeros(latent_dim)
        }
        
        # Decoder network parameters
        self.decoder_weights = {
            'W1': np.random.randn(latent_dim, 128) * np.sqrt(2.0 / latent_dim),
            'b1': np.zeros(128),
            'W2': np.random.randn(128, input_dim) * np.sqrt(2.0 / 128),
            'b2': np.zeros(input_dim)
        }
        
    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode input to latent distribution parameters.
        
        VAEs encode to distribution (mean, variance) not point estimate.
        This allows modeling uncertainty in latent representation.
        
        Args:
            x: Input data (batch_size, input_dim)
            
        Returns:
            (mu, log_var): Mean and log-variance of latent distribution
        """
        # Hidden layer with ReLU
        h = np.maximum(0, np.dot(x, self.encoder_weights['W1']) + self.encoder_weights['b1'])
        
        # Output distribution parameters
        # Mean of latent distribution
        mu = np.dot(h, self.encoder_weights['W_mu']) + self.encoder_weights['b_mu']
        
        # Log-variance (more stable than variance directly)
        log_var = np.dot(h, self.encoder_weights['W_logvar']) + self.encoder_weights['b_logvar']
        
        return mu, log_var
    
    def reparameterization_trick(self, mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
        """
        Sample from latent distribution using reparameterization trick.
        
        Key insight: We can't backpropagate through random sampling,
        so we reparameterize: z = μ + σ * ε, where ε ~ N(0,1)
        
        This makes sampling differentiable with respect to parameters.
        
        Args:
            mu: Mean of latent distribution
            log_var: Log-variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        # Standard deviation from log-variance
        std = np.exp(0.5 * log_var)
        
        # Sample from standard normal
        epsilon = np.random.normal(0, 1, mu.shape)
        
        # Reparameterize
        z = mu + std * epsilon
        
        return z
    
    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        Decode latent vector to reconstruction.
        
        Args:
            z: Latent vector (batch_size, latent_dim)
            
        Returns:
            Reconstructed data
        """
        # Hidden layer with ReLU
        h = np.maximum(0, np.dot(z, self.decoder_weights['W1']) + self.decoder_weights['b1'])
        
        # Output layer (sigmoid for bounded output)
        x_reconstructed = 1.0 / (1.0 + np.exp(
            -np.dot(h, self.decoder_weights['W2']) - self.decoder_weights['b2']
        ))
        
        return x_reconstructed
    
    def compute_loss(self, x: np.ndarray, x_reconstructed: np.ndarray,
                    mu: np.ndarray, log_var: np.ndarray) -> Dict[str, float]:
        """
        Compute VAE loss (ELBO = reconstruction loss + KL divergence).
        
        ELBO = E[log p(x|z)] - KL[q(z|x) || p(z)]
        
        Where:
        - E[log p(x|z)]: Reconstruction loss (how well we reconstruct)
        - KL[q(z|x) || p(z)]: KL divergence (how close latent dist is to prior)
        
        Args:
            x: Original input
            x_reconstructed: Reconstructed input
            mu: Latent mean
            log_var: Latent log-variance
            
        Returns:
            Dictionary with loss components
        """
        # Reconstruction loss (binary cross-entropy for bounded data)
        # Could also use MSE for continuous data
        reconstruction_loss = -np.sum(
            x * np.log(x_reconstructed + 1e-8) + 
            (1 - x) * np.log(1 - x_reconstructed + 1e-8)
        )
        
        # KL divergence from N(0,1) prior
        # Analytical formula for KL between two Gaussians
        # KL = -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))
        
        # Total loss (ELBO = negative of this)
        total_loss = reconstruction_loss + kl_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'kl_loss': kl_loss
        }


class DiffusionModel:
    """
    Denoising Diffusion Probabilistic Model for high-quality generation.
    
    Use when: Need diverse, high-quality generation with stable training
    Similar to: GANs (but more stable), Score Matching, VAEs
    
    Key insight: Gradually add noise (forward process) then learn to reverse it
    (reverse process), enabling controlled generation.
    """
    
    def __init__(self, data_dim: int, num_timesteps: int = 1000):
        """
        Initialize diffusion model.
        
        Args:
            data_dim: Dimension of data
            num_timesteps: Number of diffusion steps (more = better quality but slower)
        """
        self.data_dim = data_dim
        self.num_timesteps = num_timesteps
        
        # Define noise schedule (linear schedule is simplest)
        # Beta values control how much noise is added at each step
        self.betas = np.linspace(1e-4, 0.02, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        
        # Precompute values for efficiency
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        
    def forward_diffusion(self, x_0: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward diffusion process: Add noise to data.
        
        Using the nice property that we can sample x_t directly:
        x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε, where ε ~ N(0, I)
        
        This avoids having to simulate all intermediate steps.
        
        Args:
            x_0: Original data
            t: Timestep (0 to num_timesteps-1)
            
        Returns:
            (x_t, noise): Noisy data and noise added
        """
        # Sample noise
        noise = np.random.normal(0, 1, x_0.shape)
        
        # Add noise according to schedule
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Direct sampling formula
        x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
        
        return x_t, noise
    
    def predict_noise(self, x_t: np.ndarray, t: int) -> np.ndarray:
        """
        Neural network to predict noise (would be learned in practice).
        
        This is the core of diffusion models: learning to predict the noise
        that was added, which allows us to reverse the process.
        
        Args:
            x_t: Noisy data at timestep t
            t: Current timestep
            
        Returns:
            Predicted noise
        """
        # Simplified: In practice, this would be a U-Net or similar architecture
        # that takes both x_t and t as input
        
        # For demo, return random noise (in practice, this is learned)
        return np.random.normal(0, 1, x_t.shape)
    
    def reverse_diffusion_step(self, x_t: np.ndarray, t: int, 
                              predicted_noise: np.ndarray) -> np.ndarray:
        """
        Single reverse diffusion step: Remove noise.
        
        Uses the formula:
        x_{t-1} = 1/√α_t * (x_t - (1-α_t)/√(1-ᾱ_t) * predicted_noise) + σ_t * z
        
        Args:
            x_t: Noisy data at timestep t
            t: Current timestep
            predicted_noise: Predicted noise from neural network
            
        Returns:
            Less noisy data x_{t-1}
        """
        # Get parameters for this timestep
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]
        
        # Compute x_{t-1} mean
        x_t_minus_1_mean = (1.0 / np.sqrt(alpha_t)) * (
            x_t - (beta_t / self.sqrt_one_minus_alphas_cumprod[t]) * predicted_noise
        )
        
        # Add noise (except at t=0)
        if t > 0:
            # Variance for reverse step
            sigma_t = np.sqrt(beta_t)
            noise = np.random.normal(0, 1, x_t.shape)
            x_t_minus_1 = x_t_minus_1_mean + sigma_t * noise
        else:
            x_t_minus_1 = x_t_minus_1_mean
            
        return x_t_minus_1
    
    def generate(self, shape: Tuple[int, ...]) -> np.ndarray:
        """
        Generate new data by reversing diffusion from pure noise.
        
        Start from random noise and gradually denoise to generate data.
        
        Args:
            shape: Shape of data to generate
            
        Returns:
            Generated data
        """
        # Start from pure noise
        x = np.random.normal(0, 1, shape)
        
        # Reverse diffusion process
        for t in reversed(range(self.num_timesteps)):
            # Predict noise at this timestep
            predicted_noise = self.predict_noise(x, t)
            
            # Remove noise
            x = self.reverse_diffusion_step(x, t, predicted_noise)
            
        return x


class QLearningAgent:
    """
    Q-Learning for value-based reinforcement learning.
    
    Use when: Discrete actions, small/tabular state space, need optimal policy
    Similar to: DQN (Deep Q-Network), Policy Gradient methods
    
    Key insight: Learn action-value function Q(s,a) that estimates expected
    future reward, then act greedily with respect to Q.
    """
    
    def __init__(self, n_states: int, n_actions: int, learning_rate: float = 0.1,
                 discount_factor: float = 0.99, epsilon: float = 0.1):
        """
        Initialize Q-learning agent.
        
        Args:
            n_states: Number of states in environment
            n_actions: Number of possible actions
            learning_rate: Step size for Q-value updates (α)
            discount_factor: Importance of future rewards (γ)
            epsilon: Exploration rate for ε-greedy policy
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        # Initialize Q-table with small random values
        # Better than zeros - encourages exploration
        self.q_table = np.random.uniform(-0.01, 0.01, (n_states, n_actions))
        
        # Track visit counts for UCB exploration (advanced technique)
        self.state_action_counts = np.zeros((n_states, n_actions))
        
    def select_action(self, state: int, training: bool = True) -> int:
        """
        Select action using ε-greedy policy.
        
        Balances exploration vs exploitation:
        - Exploration: Try random actions to discover better strategies
        - Exploitation: Use current knowledge to maximize reward
        
        Args:
            state: Current state
            training: Whether in training mode (use exploration)
            
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: choose best action according to Q-values
            # Break ties randomly (important for early training)
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            
            # Get all actions with maximum Q-value
            best_actions = np.where(q_values == max_q)[0]
            
            # Randomly select among best actions
            return np.random.choice(best_actions)
    
    def update(self, state: int, action: int, reward: float, 
              next_state: int, done: bool) -> float:
        """
        Update Q-value using Bellman equation.
        
        Q-learning update rule:
        Q(s,a) ← Q(s,a) + α[r + γ*max_a' Q(s',a') - Q(s,a)]
        
        This is off-policy: we update using max Q-value (greedy policy)
        regardless of how we actually selected the action.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode ended
            
        Returns:
            TD error (for debugging/monitoring)
        """
        # Current Q-value
        current_q = self.q_table[state, action]
        
        # Target Q-value
        if done:
            # No future rewards if episode ended
            target_q = reward
        else:
            # Bootstrap from next state's value
            # max_a' Q(s', a') - this is what makes it Q-learning
            target_q = reward + self.gamma * np.max(self.q_table[next_state])
            
        # Temporal Difference (TD) error
        td_error = target_q - current_q
        
        # Update Q-value
        self.q_table[state, action] += self.lr * td_error
        
        # Update visit counts
        self.state_action_counts[state, action] += 1
        
        return td_error
    
    def decay_epsilon(self, decay_rate: float = 0.995):
        """
        Decay exploration rate over time.
        
        Start with high exploration, gradually shift to exploitation
        as we learn more about the environment.
        """
        self.epsilon = max(0.01, self.epsilon * decay_rate)


class ActorCriticAgent:
    """
    Actor-Critic for continuous action spaces and policy gradient.
    
    Use when: Continuous actions, high-dimensional state/action spaces
    Similar to: PPO, A3C, SAC, TD3
    
    Key insight: Combine value function (critic) with policy (actor)
    to reduce variance in policy gradient estimates.
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        """
        Initialize Actor-Critic architecture.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Actor (policy) network parameters
        self.actor_weights = {
            'W1': np.random.randn(state_dim, 64) * np.sqrt(2.0 / state_dim),
            'b1': np.zeros(64),
            'W_mu': np.random.randn(64, action_dim) * np.sqrt(2.0 / 64),
            'b_mu': np.zeros(action_dim),
            'W_sigma': np.random.randn(64, action_dim) * np.sqrt(2.0 / 64),
            'b_sigma': np.zeros(action_dim)
        }
        
        # Critic (value function) network parameters
        self.critic_weights = {
            'W1': np.random.randn(state_dim, 64) * np.sqrt(2.0 / state_dim),
            'b1': np.zeros(64),
            'W2': np.random.randn(64, 1) * np.sqrt(2.0 / 64),
            'b2': np.zeros(1)
        }
        
    def actor_forward(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through actor network to get action distribution.
        
        Outputs parameters of Gaussian distribution for continuous actions.
        
        Args:
            state: Current state
            
        Returns:
            (mu, sigma): Mean and standard deviation of action distribution
        """
        # Hidden layer with ReLU
        h = np.maximum(0, np.dot(state, self.actor_weights['W1']) + self.actor_weights['b1'])
        
        # Output mean of action distribution
        mu = np.dot(h, self.actor_weights['W_mu']) + self.actor_weights['b_mu']
        
        # Output log std (more stable than std directly)
        log_sigma = np.dot(h, self.actor_weights['W_sigma']) + self.actor_weights['b_sigma']
        
        # Convert to std with minimum value for stability
        sigma = np.exp(log_sigma) + 1e-5
        
        return mu, sigma
    
    def critic_forward(self, state: np.ndarray) -> float:
        """
        Forward pass through critic network to estimate state value.
        
        Args:
            state: Current state
            
        Returns:
            Estimated value of state
        """
        # Hidden layer with ReLU
        h = np.maximum(0, np.dot(state, self.critic_weights['W1']) + self.critic_weights['b1'])
        
        # Output state value
        value = np.dot(h, self.critic_weights['W2']) + self.critic_weights['b2']
        
        return value.squeeze()
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Sample action from policy distribution.
        
        Args:
            state: Current state
            
        Returns:
            Sampled action
        """
        mu, sigma = self.actor_forward(state)
        
        # Sample from Gaussian distribution
        action = np.random.normal(mu, sigma)
        
        return action
    
    def compute_advantage(self, reward: float, state_value: float, 
                         next_state_value: float, done: bool, gamma: float = 0.99) -> float:
        """
        Compute advantage function A(s,a) = Q(s,a) - V(s).
        
        Uses TD error as advantage estimate:
        A = r + γV(s') - V(s)
        
        Advantage tells us how much better an action is compared to average.
        
        Args:
            reward: Immediate reward
            state_value: Value of current state
            next_state_value: Value of next state
            done: Whether episode ended
            gamma: Discount factor
            
        Returns:
            Advantage estimate
        """
        if done:
            target_value = reward
        else:
            target_value = reward + gamma * next_state_value
            
        advantage = target_value - state_value
        
        return advantage


class DBSCANClustering:
    """
    DBSCAN: Density-Based Spatial Clustering of Applications with Noise.
    
    Use when: Arbitrary cluster shapes, outlier detection needed, density varies
    Similar to: OPTICS, HDBSCAN, Mean Shift
    
    Key insight: Clusters are dense regions separated by sparse regions.
    Can find non-convex clusters and identify outliers.
    """
    
    def __init__(self, eps: float = 0.5, min_points: int = 5):
        """
        Initialize DBSCAN parameters.
        
        Args:
            eps: Maximum distance between two points to be neighbors
            min_points: Minimum points to form a dense region (core point)
        """
        self.eps = eps
        self.min_points = min_points
        self.labels_ = None
        self.core_points_ = set()
        
    def fit(self, X: np.ndarray) -> np.ndarray:
        """
        Perform DBSCAN clustering.
        
        Algorithm:
        1. Find core points (points with >= min_points neighbors)
        2. Build clusters from core points and their neighborhoods
        3. Assign border points to clusters
        4. Mark remaining points as noise
        
        Args:
            X: Data points (n_samples, n_features)
            
        Returns:
            Cluster labels (-1 for noise)
        """
        n_samples = X.shape[0]
        
        # Compute pairwise distances
        # In production, use spatial index (KD-tree) for efficiency
        distances = cdist(X, X)
        
        # Find neighborhoods for each point
        neighborhoods = []
        for i in range(n_samples):
            # Points within eps distance
            neighbors = np.where(distances[i] <= self.eps)[0]
            neighborhoods.append(neighbors)
            
            # Mark core points
            if len(neighbors) >= self.min_points:
                self.core_points_.add(i)
                
        # Initialize labels (-1 means unassigned/noise)
        labels = np.full(n_samples, -1)
        cluster_id = 0
        
        # Build clusters from core points
        for point in range(n_samples):
            # Skip if already processed
            if labels[point] != -1:
                continue
                
            # Skip if not core point
            if point not in self.core_points_:
                continue
                
            # Start new cluster
            labels[point] = cluster_id
            
            # Expand cluster using BFS
            queue = deque(neighborhoods[point])
            processed = {point}
            
            while queue:
                neighbor = queue.popleft()
                
                if neighbor in processed:
                    continue
                    
                processed.add(neighbor)
                
                # Assign to cluster if not noise
                if labels[neighbor] == -1:
                    labels[neighbor] = cluster_id
                    
                # If neighbor is core point, add its neighbors
                if neighbor in self.core_points_:
                    for nn in neighborhoods[neighbor]:
                        if nn not in processed:
                            queue.append(nn)
                            
            cluster_id += 1
            
        self.labels_ = labels
        return labels
    
    def predict(self, X_new: np.ndarray, X_train: np.ndarray) -> np.ndarray:
        """
        Assign new points to existing clusters.
        
        Args:
            X_new: New data points
            X_train: Original training data
            
        Returns:
            Cluster labels for new points
        """
        n_new = X_new.shape[0]
        labels_new = np.full(n_new, -1)
        
        for i in range(n_new):
            # Find distances to training points
            dists = cdist([X_new[i]], X_train)[0]
            
            # Find neighbors within eps
            neighbors = np.where(dists <= self.eps)[0]
            
            if len(neighbors) > 0:
                # Get labels of neighbors (excluding noise)
                neighbor_labels = self.labels_[neighbors]
                valid_labels = neighbor_labels[neighbor_labels != -1]
                
                if len(valid_labels) > 0:
                    # Assign to most common cluster among neighbors
                    unique, counts = np.unique(valid_labels, return_counts=True)
                    labels_new[i] = unique[np.argmax(counts)]
                    
        return labels_new


class SpectralClustering:
    """
    Spectral Clustering using graph Laplacian eigenvectors.
    
    Use when: Non-convex clusters, manifold structure, graph/network data
    Similar to: Graph Neural Networks, Laplacian Eigenmaps, Diffusion Maps
    
    Key insight: Project data onto eigenvectors of graph Laplacian before
    clustering, revealing global structure through local connectivity.
    """
    
    def __init__(self, n_clusters: int, affinity: str = 'rbf', gamma: float = 1.0):
        """
        Initialize spectral clustering.
        
        Args:
            n_clusters: Number of clusters
            affinity: Affinity function ('rbf' for RBF kernel)
            gamma: Kernel coefficient for RBF
        """
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
        
    def construct_affinity_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Construct affinity (similarity) matrix.
        
        Using RBF kernel: A_ij = exp(-γ ||x_i - x_j||²)
        
        Args:
            X: Data points (n_samples, n_features)
            
        Returns:
            Affinity matrix (n_samples, n_samples)
        """
        # Compute pairwise squared Euclidean distances
        pairwise_dists_sq = cdist(X, X, 'sqeuclidean')
        
        # Apply RBF kernel
        if self.affinity == 'rbf':
            affinity_matrix = np.exp(-self.gamma * pairwise_dists_sq)
        else:
            # Simple binary affinity based on k-nearest neighbors
            k = 10
            affinity_matrix = np.zeros_like(pairwise_dists_sq)
            for i in range(len(X)):
                # Find k nearest neighbors
                neighbors = np.argsort(pairwise_dists_sq[i])[:k+1]
                affinity_matrix[i, neighbors] = 1
                affinity_matrix[neighbors, i] = 1  # Symmetrize
                
        return affinity_matrix
    
    def compute_laplacian(self, affinity_matrix: np.ndarray, 
                         normalized: bool = True) -> np.ndarray:
        """
        Compute graph Laplacian matrix.
        
        Laplacian encodes graph structure:
        - Unnormalized: L = D - A
        - Normalized: L = I - D^(-1/2) A D^(-1/2)
        
        Args:
            affinity_matrix: Affinity/adjacency matrix
            normalized: Whether to use normalized Laplacian
            
        Returns:
            Laplacian matrix
        """
        # Degree matrix (diagonal matrix of row sums)
        degree = np.sum(affinity_matrix, axis=1)
        degree_matrix = np.diag(degree)
        
        if normalized:
            # Normalized Laplacian (better for clusters of different sizes)
            # Avoid division by zero
            degree_sqrt_inv = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-10)))
            
            # L = I - D^(-1/2) A D^(-1/2)
            normalized_affinity = degree_sqrt_inv @ affinity_matrix @ degree_sqrt_inv
            laplacian = np.eye(len(affinity_matrix)) - normalized_affinity
        else:
            # Unnormalized Laplacian
            laplacian = degree_matrix - affinity_matrix
            
        return laplacian
    
    def fit(self, X: np.ndarray) -> np.ndarray:
        """
        Perform spectral clustering.
        
        Algorithm:
        1. Construct affinity matrix
        2. Compute graph Laplacian
        3. Find eigenvectors of Laplacian
        4. Cluster eigenvector representation with k-means
        
        Args:
            X: Data points (n_samples, n_features)
            
        Returns:
            Cluster labels
        """
        # Step 1: Construct affinity matrix
        affinity_matrix = self.construct_affinity_matrix(X)
        
        # Step 2: Compute normalized Laplacian
        laplacian = self.compute_laplacian(affinity_matrix)
        
        # Step 3: Eigendecomposition
        # Find smallest eigenvalues/eigenvectors
        # Using scipy for efficiency with sparse matrices
        eigenvalues, eigenvectors = eigsh(laplacian, k=self.n_clusters, which='SM')
        
        # Step 4: Normalize eigenvectors (rows to unit length)
        # This is the spectral embedding
        embedding = eigenvectors / np.linalg.norm(eigenvectors, axis=1, keepdims=True)
        
        # Step 5: Cluster in embedding space using k-means
        labels = self.kmeans_clustering(embedding, self.n_clusters)
        
        return labels
    
    def kmeans_clustering(self, X: np.ndarray, k: int, max_iters: int = 100) -> np.ndarray:
        """
        Simple k-means implementation for clustering embedded points.
        
        Args:
            X: Points to cluster
            k: Number of clusters
            max_iters: Maximum iterations
            
        Returns:
            Cluster labels
        """
        n_samples = X.shape[0]
        
        # Initialize centroids with k-means++
        centroids = self.kmeans_plusplus_init(X, k)
        
        labels = np.zeros(n_samples, dtype=int)
        
        for _ in range(max_iters):
            # Assign points to nearest centroid
            old_labels = labels.copy()
            
            for i in range(n_samples):
                distances = np.linalg.norm(X[i] - centroids, axis=1)
                labels[i] = np.argmin(distances)
                
            # Update centroids
            for j in range(k):
                cluster_points = X[labels == j]
                if len(cluster_points) > 0:
                    centroids[j] = np.mean(cluster_points, axis=0)
                    
            # Check convergence
            if np.array_equal(labels, old_labels):
                break
                
        return labels
    
    def kmeans_plusplus_init(self, X: np.ndarray, k: int) -> np.ndarray:
        """
        K-means++ initialization for better convergence.
        
        Selects initial centers with probability proportional to squared
        distance from nearest center, spreading them out.
        
        Args:
            X: Data points
            k: Number of centers
            
        Returns:
            Initial centroids
        """
        n_samples = X.shape[0]
        centroids = []
        
        # Choose first center randomly
        first_idx = np.random.randint(n_samples)
        centroids.append(X[first_idx])
        
        for _ in range(1, k):
            # Compute distances to nearest centroid
            distances = np.full(n_samples, np.inf)
            
            for centroid in centroids:
                dists = np.linalg.norm(X - centroid, axis=1)
                distances = np.minimum(distances, dists)
                
            # Choose next centroid with probability proportional to squared distance
            probabilities = distances ** 2
            probabilities /= probabilities.sum()
            
            next_idx = np.random.choice(n_samples, p=probabilities)
            centroids.append(X[next_idx])
            
        return np.array(centroids)


class GraphNeuralNetwork:
    """
    Graph Neural Network for learning on graph-structured data.
    
    Use when: Data has explicit relational structure (social networks, molecules)
    Similar to: CNNs (but for graphs), Graph Transformers, Spectral methods
    
    Key insight: Aggregate information from neighbors to update node representations,
    capturing both node features and graph topology.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Initialize GNN with message passing.
        
        Args:
            input_dim: Dimension of node features
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension per node
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Message passing weights
        self.W_message = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.W_update = np.random.randn(hidden_dim * 2, hidden_dim) * np.sqrt(2.0 / (hidden_dim * 2))
        self.W_output = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        
    def message_passing_layer(self, node_features: np.ndarray, 
                            adjacency: np.ndarray) -> np.ndarray:
        """
        Single message passing layer (Graph Convolution).
        
        Each node aggregates messages from neighbors and updates its representation.
        
        Formula: h_i' = σ(W_update * [h_i || AGG({W_message * h_j : j ∈ N(i)})])
        
        Args:
            node_features: Node feature matrix (n_nodes, feature_dim)
            adjacency: Adjacency matrix (n_nodes, n_nodes)
            
        Returns:
            Updated node features
        """
        n_nodes = node_features.shape[0]
        
        # Transform node features for messages
        messages = np.dot(node_features, self.W_message)
        
        # Aggregate messages from neighbors (mean aggregation)
        # This could also be sum, max, or learned aggregation
        aggregated = np.zeros((n_nodes, self.hidden_dim))
        
        for i in range(n_nodes):
            # Find neighbors
            neighbors = np.where(adjacency[i] > 0)[0]
            
            if len(neighbors) > 0:
                # Average messages from neighbors
                neighbor_messages = messages[neighbors]
                aggregated[i] = np.mean(neighbor_messages, axis=0)
            else:
                # No neighbors - use zero vector
                aggregated[i] = np.zeros(self.hidden_dim)
                
        # Combine node's own features with aggregated messages
        combined = np.concatenate([messages, aggregated], axis=1)
        
        # Update node representations
        updated = np.maximum(0, np.dot(combined, self.W_update))  # ReLU activation
        
        return updated
    
    def forward(self, node_features: np.ndarray, adjacency: np.ndarray, 
               num_layers: int = 2) -> np.ndarray:
        """
        Forward pass through multiple GNN layers.
        
        Stacking layers allows information to propagate further in graph.
        With k layers, each node aggregates information from k-hop neighborhood.
        
        Args:
            node_features: Initial node features (n_nodes, input_dim)
            adjacency: Adjacency matrix
            num_layers: Number of message passing layers
            
        Returns:
            Final node representations (n_nodes, output_dim)
        """
        h = node_features
        
        # Apply message passing layers
        for _ in range(num_layers):
            h = self.message_passing_layer(h, adjacency)
            
        # Final output transformation
        output = np.dot(h, self.W_output)
        
        return output
    
    def graph_pooling(self, node_features: np.ndarray, 
                     pooling: str = 'mean') -> np.ndarray:
        """
        Pool node features to get graph-level representation.
        
        Args:
            node_features: Node feature matrix
            pooling: Pooling strategy ('mean', 'max', 'sum')
            
        Returns:
            Graph-level feature vector
        """
        if pooling == 'mean':
            return np.mean(node_features, axis=0)
        elif pooling == 'max':
            return np.max(node_features, axis=0)
        elif pooling == 'sum':
            return np.sum(node_features, axis=0)
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")


class GradientBoostedTrees:
    """
    Simplified Gradient Boosted Decision Trees for tabular data.
    
    Use when: Tabular/structured data, need high accuracy, interpretability matters
    Similar to: XGBoost, LightGBM, CatBoost, Random Forests
    
    Key insight: Sequentially fit trees to residuals, each tree corrects
    errors of previous ensemble.
    """
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3):
        """
        Initialize GBDT.
        
        Args:
            n_estimators: Number of boosting rounds (trees)
            learning_rate: Shrinkage parameter to prevent overfitting
            max_depth: Maximum tree depth
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train gradient boosted trees.
        
        Algorithm:
        1. Initialize with constant prediction
        2. For each round:
           - Compute residuals (negative gradients)
           - Fit tree to residuals
           - Add tree to ensemble with learning rate
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)
        """
        n_samples = X.shape[0]
        
        # Initialize with mean (for regression)
        self.initial_prediction = np.mean(y)
        predictions = np.full(n_samples, self.initial_prediction)
        
        for i in range(self.n_estimators):
            # Compute residuals (negative gradient of squared loss)
            residuals = y - predictions
            
            # Fit decision tree to residuals
            tree = self._fit_tree(X, residuals, self.max_depth)
            self.trees.append(tree)
            
            # Update predictions
            tree_predictions = self._predict_tree(tree, X)
            predictions += self.learning_rate * tree_predictions
            
            # Calculate training loss for monitoring
            mse = np.mean((y - predictions) ** 2)
            
            if (i + 1) % 10 == 0:
                print(f"Round {i + 1}/{self.n_estimators}, MSE: {mse:.4f}")
                
    def _fit_tree(self, X: np.ndarray, y: np.ndarray, max_depth: int) -> Dict:
        """
        Fit single decision tree (simplified CART algorithm).
        
        This is a basic implementation. Production versions use:
        - Histogram-based splitting (faster)
        - Regularization (min_samples_split, min_child_weight)
        - Column subsampling
        - Missing value handling
        
        Args:
            X: Features
            y: Targets
            max_depth: Maximum depth
            
        Returns:
            Tree structure as dictionary
        """
        def build_tree(X_subset, y_subset, depth):
            n_samples = len(y_subset)
            
            # Stopping criteria
            if depth >= max_depth or n_samples < 10:
                # Leaf node: return mean value
                return {'type': 'leaf', 'value': np.mean(y_subset)}
                
            # Find best split
            best_gain = 0
            best_split = None
            
            # Try each feature
            for feature in range(X.shape[1]):
                # Sort by feature value
                sorted_indices = np.argsort(X_subset[:, feature])
                
                # Try each split point
                for i in range(1, n_samples):
                    if X_subset[sorted_indices[i-1], feature] == X_subset[sorted_indices[i], feature]:
                        continue  # Same value, can't split
                        
                    # Split threshold
                    threshold = (X_subset[sorted_indices[i-1], feature] + 
                               X_subset[sorted_indices[i], feature]) / 2
                    
                    # Split data
                    left_mask = X_subset[:, feature] <= threshold
                    right_mask = ~left_mask
                    
                    if np.sum(left_mask) < 5 or np.sum(right_mask) < 5:
                        continue  # Too few samples in split
                        
                    # Calculate variance reduction (simplified gain)
                    var_parent = np.var(y_subset)
                    var_left = np.var(y_subset[left_mask])
                    var_right = np.var(y_subset[right_mask])
                    
                    n_left = np.sum(left_mask)
                    n_right = np.sum(right_mask)
                    
                    # Information gain (variance reduction)
                    gain = var_parent - (n_left/n_samples * var_left + 
                                        n_right/n_samples * var_right)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_split = {
                            'feature': feature,
                            'threshold': threshold,
                            'left_indices': np.where(left_mask)[0],
                            'right_indices': np.where(right_mask)[0]
                        }
            
            # If no good split found, make leaf
            if best_split is None:
                return {'type': 'leaf', 'value': np.mean(y_subset)}
            
            # Recursively build subtrees
            left_tree = build_tree(
                X_subset[best_split['left_indices']], 
                y_subset[best_split['left_indices']], 
                depth + 1
            )
            right_tree = build_tree(
                X_subset[best_split['right_indices']], 
                y_subset[best_split['right_indices']], 
                depth + 1
            )
            
            return {
                'type': 'split',
                'feature': best_split['feature'],
                'threshold': best_split['threshold'],
                'left': left_tree,
                'right': right_tree
            }
        
        return build_tree(X, y, 0)
    
    def _predict_tree(self, tree: Dict, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using single tree.
        
        Args:
            tree: Tree structure
            X: Features to predict
            
        Returns:
            Predictions for each sample
        """
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        for i in range(n_samples):
            node = tree
            
            # Traverse tree until leaf
            while node['type'] == 'split':
                if X[i, node['feature']] <= node['threshold']:
                    node = node['left']
                else:
                    node = node['right']
                    
            predictions[i] = node['value']
            
        return predictions
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using ensemble.
        
        Args:
            X: Features to predict
            
        Returns:
            Ensemble predictions
        """
        predictions = np.full(X.shape[0], self.initial_prediction)
        
        for tree in self.trees:
            predictions += self.learning_rate * self._predict_tree(tree, X)
            
        return predictions
    
    def feature_importance(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate feature importance based on splits.
        
        Returns:
            Importance score for each feature
        """
        n_features = X.shape[1]
        importance = np.zeros(n_features)
        
        for tree in self.trees:
            self._accumulate_importance(tree, importance)
            
        # Normalize
        importance = importance / importance.sum()
        
        return importance
    
    def _accumulate_importance(self, node: Dict, importance: np.ndarray):
        """Recursively accumulate feature importance from tree."""
        if node['type'] == 'split':
            importance[node['feature']] += 1
            self._accumulate_importance(node['left'], importance)
            self._accumulate_importance(node['right'], importance)


class MatrixFactorizationRecommender:
    """
    Matrix Factorization for collaborative filtering recommender systems.
    
    Use when: Sparse user-item matrices, implicit/explicit feedback
    Similar to: SVD, ALS, Neural Collaborative Filtering
    
    Key insight: Decompose user-item matrix into low-rank user and item
    embeddings that capture latent preferences/characteristics.
    """
    
    def __init__(self, n_factors: int = 50, learning_rate: float = 0.01,
                 regularization: float = 0.01):
        """
        Initialize matrix factorization model.
        
        Args:
            n_factors: Number of latent factors (embedding dimension)
            learning_rate: SGD learning rate
            regularization: L2 regularization strength
        """
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = regularization
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = None
        
    def fit(self, ratings: np.ndarray, n_epochs: int = 50):
        """
        Train matrix factorization using SGD.
        
        Optimizes: min Σ(r_ui - μ - b_u - b_i - p_u·q_i)² + λ(||p_u||² + ||q_i||² + b_u² + b_i²)
        
        Where:
        - r_ui: observed rating
        - μ: global mean
        - b_u, b_i: user and item biases
        - p_u, q_i: user and item latent factors
        
        Args:
            ratings: Sparse rating matrix (n_users, n_items), 0 means missing
            n_epochs: Number of training epochs
        """
        n_users, n_items = ratings.shape
        
        # Initialize with small random values
        self.user_factors = np.random.normal(0, 0.01, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.01, (n_items, self.n_factors))
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        
        # Compute global mean of observed ratings
        mask = ratings > 0
        self.global_mean = ratings[mask].mean()
        
        # Get list of observed ratings for efficient iteration
        observed_ratings = []
        for u in range(n_users):
            for i in range(n_items):
                if ratings[u, i] > 0:
                    observed_ratings.append((u, i, ratings[u, i]))
        
        # SGD training
        for epoch in range(n_epochs):
            # Shuffle for SGD
            np.random.shuffle(observed_ratings)
            
            total_error = 0
            
            for u, i, r in observed_ratings:
                # Predict rating
                prediction = self._predict_single(u, i)
                
                # Compute error
                error = r - prediction
                total_error += error ** 2
                
                # SGD updates with regularization
                # Update biases
                self.user_biases[u] += self.lr * (error - self.reg * self.user_biases[u])
                self.item_biases[i] += self.lr * (error - self.reg * self.item_biases[i])
                
                # Update latent factors
                user_factors_u = self.user_factors[u].copy()
                
                self.user_factors[u] += self.lr * (
                    error * self.item_factors[i] - self.reg * self.user_factors[u]
                )
                self.item_factors[i] += self.lr * (
                    error * user_factors_u - self.reg * self.item_factors[i]
                )
            
            # Calculate RMSE
            rmse = np.sqrt(total_error / len(observed_ratings))
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}, RMSE: {rmse:.4f}")
    
    def _predict_single(self, user: int, item: int) -> float:
        """
        Predict single rating.
        
        Args:
            user: User index
            item: Item index
            
        Returns:
            Predicted rating
        """
        prediction = (self.global_mean + 
                     self.user_biases[user] + 
                     self.item_biases[item] + 
                     np.dot(self.user_factors[user], self.item_factors[item]))
        
        # Clip to valid rating range (e.g., 1-5)
        return np.clip(prediction, 1, 5)
    
    def predict(self, user: int, item: int) -> float:
        """
        Predict rating for user-item pair.
        
        Args:
            user: User index
            item: Item index
            
        Returns:
            Predicted rating
        """
        if self.user_factors is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        return self._predict_single(user, item)
    
    def recommend_items(self, user: int, n_recommendations: int = 10,
                        exclude_seen: bool = True, seen_items: np.ndarray = None) -> np.ndarray:
        """
        Recommend top-N items for user.
        
        Args:
            user: User index
            n_recommendations: Number of items to recommend
            exclude_seen: Whether to exclude already rated items
            seen_items: Items already seen by user
            
        Returns:
            Indices of recommended items
        """
        n_items = self.item_factors.shape[0]
        
        # Predict ratings for all items
        predictions = []
        for item in range(n_items):
            if exclude_seen and seen_items is not None and item in seen_items:
                continue
                
            pred = self._predict_single(user, item)
            predictions.append((item, pred))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N
        recommendations = [item for item, _ in predictions[:n_recommendations]]
        
        return np.array(recommendations)


def run_comprehensive_demo():
    """
    Comprehensive demonstration of all algorithms with real use cases.
    Shows when to use each algorithm and interprets results.
    """
    print("COMPREHENSIVE ADVANCED ALGORITHMS DEMONSTRATION")
    print("=" * 80)
    
    np.random.seed(42)  # For reproducibility
    
    # 1. CNN for Image Feature Extraction
    print("\n1. CNN FOR SPATIAL REPRESENTATION LEARNING")
    print("-" * 40)
    print("Use case: Extract features from image data")
    print("When to use: Grid-like data with local patterns (images, spectrograms)")
    
    # Create synthetic image (28x28 grayscale)
    image = np.random.rand(28, 28, 1)
    cnn = CNNFeatureExtractor(input_shape=(28, 28, 1), num_filters=16)
    features = cnn.forward(image)
    
    print(f"Input image shape: {image.shape}")
    print(f"Extracted features shape: {features.shape}")
    print(f"Features capture spatial patterns through convolution and pooling")
    print("Similar alternatives: Vision Transformers (ViT) for global context")
    
    # 2. LSTM for Sequence Modeling
    print("\n2. LSTM FOR SEQUENCE MODELING WITH MEMORY")
    print("-" * 40)
    print("Use case: Time series prediction, language modeling")
    print("When to use: Sequential data with long-term dependencies")
    
    # Create synthetic sequence
    sequence_length = 20
    input_dim = 10
    sequence = np.random.randn(sequence_length, input_dim)
    
    lstm = LSTMSequenceModel(input_dim=input_dim, hidden_dim=32)
    hidden_states = lstm.forward(sequence)
    
    print(f"Input sequence shape: {sequence.shape}")
    print(f"Hidden states shape: {hidden_states.shape}")
    print("LSTM maintains memory across timesteps via gating mechanisms")
    print("Similar alternatives: Transformers for parallel processing, GRUs for efficiency")
    
    # 3. Transformer Self-Attention
    print("\n3. TRANSFORMER FOR ATTENTION-BASED MODELING")
    print("-" * 40)
    print("Use case: Machine translation, document understanding")
    print("When to use: Need global context, parallel processing")
    
    # Create synthetic sequence
    seq_len = 10
    d_model = 64
    sequence = np.random.randn(seq_len, d_model)
    
    transformer = SelfAttentionTransformer(d_model=d_model, num_heads=8)
    attended = transformer.multi_head_attention(sequence)
    
    print(f"Input sequence shape: {sequence.shape}")
    print(f"Attention output shape: {attended.shape}")
    print("Attention allows direct connections between any positions")
    print("Similar alternatives: RNNs for smaller sequences, sparse attention for long sequences")
    
    # 4. Variational Autoencoder
    print("\n4. VAE FOR PROBABILISTIC GENERATION")
    print("-" * 40)
    print("Use case: Generate new samples, learn latent representations")
    print("When to use: Need uncertainty estimates, interpretable latent space")
    
    # Create synthetic data
    data_dim = 100
    data = np.random.rand(10, data_dim)
    
    vae = VariationalAutoencoder(input_dim=data_dim, latent_dim=10)
    mu, log_var = vae.encode(data)
    z = vae.reparameterization_trick(mu, log_var)
    reconstructed = vae.decode(z)
    
    print(f"Original data shape: {data.shape}")
    print(f"Latent representation shape: {z.shape}")
    print(f"Reconstructed data shape: {reconstructed.shape}")
    print("VAE learns probabilistic latent space via ELBO optimization")
    print("Similar alternatives: GANs for sharper images, Normalizing Flows for exact likelihood")
    
    # 5. Diffusion Model
    print("\n5. DIFFUSION MODEL FOR HIGH-QUALITY GENERATION")
    print("-" * 40)
    print("Use case: Image synthesis, data augmentation")
    print("When to use: Need diverse, stable generation")
    
    diffusion = DiffusionModel(data_dim=50, num_timesteps=100)
    
    # Forward diffusion (add noise)
    original = np.random.randn(50)
    noisy, noise = diffusion.forward_diffusion(original, t=50)
    
    # Generate new sample
    generated = diffusion.generate(shape=(50,))
    
    print(f"Original data shape: {original.shape}")
    print(f"Noisy data (t=50) shape: {noisy.shape}")
    print(f"Generated sample shape: {generated.shape}")
    print("Diffusion models gradually denoise to generate high-quality samples")
    print("Similar alternatives: GANs for speed, VAEs for latent control")
    
    # 6. Q-Learning
    print("\n6. Q-LEARNING FOR DISCRETE ACTIONS")
    print("-" * 40)
    print("Use case: Game AI, robot navigation with discrete actions")
    print("When to use: Small state/action space, need optimal policy")
    
    q_agent = QLearningAgent(n_states=10, n_actions=4)
    
    # Simulate learning step
    state, action, reward, next_state = 0, 1, 1.0, 1
    td_error = q_agent.update(state, action, reward, next_state, done=False)
    
    print(f"Q-table shape: {q_agent.q_table.shape}")
    print(f"TD error from update: {td_error:.4f}")
    print("Q-learning learns optimal action-value function via Bellman updates")
    print("Similar alternatives: DQN for large state spaces, Policy Gradient for continuous")
    
    # 7. Actor-Critic
    print("\n7. ACTOR-CRITIC FOR CONTINUOUS CONTROL")
    print("-" * 40)
    print("Use case: Robotics, continuous control tasks")
    print("When to use: Continuous action space, high-dimensional states")
    
    ac_agent = ActorCriticAgent(state_dim=10, action_dim=2)
    
    # Sample action
    state = np.random.randn(10)
    action = ac_agent.select_action(state)
    value = ac_agent.critic_forward(state)
    
    print(f"State shape: {state.shape}")
    print(f"Action shape: {action.shape}")
    print(f"State value estimate: {value:.4f}")
    print("Actor-Critic combines policy gradient with value function")
    print("Similar alternatives: PPO for stability, SAC for sample efficiency")
    
    # 8. DBSCAN Clustering
    print("\n8. DBSCAN FOR DENSITY-BASED CLUSTERING")
    print("-" * 40)
    print("Use case: Anomaly detection, spatial clustering")
    print("When to use: Unknown number of clusters, outliers present")
    
    # Create synthetic data with clusters and noise
    from sklearn.datasets import make_moons
    X, _ = make_moons(n_samples=100, noise=0.1)
    
    dbscan = DBSCANClustering(eps=0.3, min_points=5)
    labels = dbscan.fit(X)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"Data shape: {X.shape}")
    print(f"Number of clusters found: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    print(f"Core points identified: {len(dbscan.core_points_)}")
    print("DBSCAN finds clusters of arbitrary shape and identifies outliers")
    print("Similar alternatives: HDBSCAN for varying densities, OPTICS for hierarchy")
    
    # 9. Spectral Clustering
    print("\n9. SPECTRAL CLUSTERING FOR COMPLEX GEOMETRIES")
    print("-" * 40)
    print("Use case: Image segmentation, community detection")
    print("When to use: Non-convex clusters, manifold structure")
    
    spectral = SpectralClustering(n_clusters=2, affinity='rbf', gamma=1.0)
    labels = spectral.fit(X)
    
    print(f"Data shape: {X.shape}")
    print(f"Unique clusters: {np.unique(labels)}")
    print("Spectral clustering uses graph Laplacian eigenvectors")
    print("Similar alternatives: GNNs for learned representations, Laplacian Eigenmaps")
    
    # 10. Graph Neural Network
    print("\n10. GNN FOR GRAPH-STRUCTURED DATA")
    print("-" * 40)
    print("Use case: Social networks, molecular property prediction")
    print("When to use: Data has explicit relational structure")
    
    # Create synthetic graph
    n_nodes = 5
    node_features = np.random.randn(n_nodes, 10)
    # Simple adjacency matrix (cycle graph)
    adjacency = np.eye(n_nodes, k=1) + np.eye(n_nodes, k=-1)
    adjacency[0, -1] = adjacency[-1, 0] = 1  # Close the cycle
    
    gnn = GraphNeuralNetwork(input_dim=10, hidden_dim=16, output_dim=4)
    node_embeddings = gnn.forward(node_features, adjacency)
    graph_embedding = gnn.graph_pooling(node_embeddings, 'mean')
    
    print(f"Number of nodes: {n_nodes}")
    print(f"Node features shape: {node_features.shape}")
    print(f"Node embeddings shape: {node_embeddings.shape}")
    print(f"Graph embedding shape: {graph_embedding.shape}")
    print("GNN aggregates neighbor information via message passing")
    print("Similar alternatives: Graph Transformers for attention, CNNs for grid graphs")
    
    # 11. Gradient Boosted Trees
    print("\n11. GRADIENT BOOSTED TREES FOR TABULAR DATA")
    print("-" * 40)
    print("Use case: Kaggle competitions, risk scoring")
    print("When to use: Structured/tabular data, need high accuracy")
    
    # Create synthetic tabular data
    X_train = np.random.randn(100, 5)
    y_train = X_train[:, 0] * 2 + X_train[:, 1] - X_train[:, 2] + np.random.randn(100) * 0.1
    
    gbdt = GradientBoostedTrees(n_estimators=20, learning_rate=0.1, max_depth=3)
    gbdt.fit(X_train, y_train)
    
    # Feature importance
    importance = gbdt.feature_importance(X_train)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Number of trees: {len(gbdt.trees)}")
    print(f"Feature importance: {importance}")
    print("GBDT sequentially fits trees to residuals")
    print("Similar alternatives: XGBoost/LightGBM for speed, Random Forest for parallelism")
    
    # 12. Matrix Factorization Recommender
    print("\n12. MATRIX FACTORIZATION FOR RECOMMENDATIONS")
    print("-" * 40)
    print("Use case: Movie recommendations, e-commerce")
    print("When to use: Sparse user-item interaction data")
    
    # Create synthetic ratings matrix (0 = unobserved)
    n_users, n_items = 50, 30
    ratings = np.zeros((n_users, n_items))
    # Add some random ratings
    for _ in range(200):
        u, i = np.random.randint(n_users), np.random.randint(n_items)
        ratings[u, i] = np.random.randint(1, 6)
    
    mf = MatrixFactorizationRecommender(n_factors=10, learning_rate=0.01)
    mf.fit(ratings, n_epochs=20)
    
    # Make recommendation
    user_id = 0
    seen_items = np.where(ratings[user_id] > 0)[0]
    recommendations = mf.recommend_items(user_id, n_recommendations=5, 
                                        seen_items=seen_items)
    
    print(f"Ratings matrix shape: {ratings.shape}")
    print(f"Sparsity: {(ratings == 0).sum() / ratings.size:.2%}")
    print(f"User embeddings shape: {mf.user_factors.shape}")
    print(f"Item embeddings shape: {mf.item_factors.shape}")
    print(f"Top 5 recommendations for user 0: {recommendations}")
    print("Matrix factorization learns latent user/item embeddings")
    print("Similar alternatives: Neural CF for non-linearity, Graph-based RecSys")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("\nKEY TAKEAWAYS:")
    print("1. CNNs excel at spatial/local patterns (images)")
    print("2. RNNs/LSTMs handle sequential dependencies (time series)")
    print("3. Transformers enable parallel sequence processing (NLP)")
    print("4. VAEs provide probabilistic generation with interpretable latents")
    print("5. Diffusion models offer stable, high-quality generation")
    print("6. Q-Learning solves discrete action MDPs optimally")
    print("7. Actor-Critic handles continuous control elegantly")
    print("8. DBSCAN finds arbitrary clusters and outliers")
    print("9. Spectral clustering handles complex manifold geometries")
    print("10. GNNs leverage graph structure for predictions")
    print("11. GBDTs dominate tabular/structured data tasks")
    print("12. Matrix factorization powers collaborative filtering")
    
    return True


if __name__ == "__main__":
    # Run the comprehensive demonstration
    success = run_comprehensive_demo()
    
    if success:
        print("\nAll algorithms demonstrated successfully!")
        print("Each implementation includes detailed comments explaining:")
        print("- Mathematical foundations")
        print("- When to use vs alternatives")
        print("- Key hyperparameters and their effects")
        print("- Common pitfalls and best practices")
                