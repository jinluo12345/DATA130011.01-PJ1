from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.params = {'W' : initialize_method(size=(in_dim, out_dim)), 'b' : initialize_method(size=(1, out_dim))}
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay

    @property
    def W(self):
        return self.params['W']
    
    @W.setter
    def W(self, value):
        self.params['W'] = value
    
    @property
    def b(self):
        return self.params['b']
    
    @b.setter
    def b(self, value):
        self.params['b'] = value  
             
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        output = np.matmul(X, self.W) + self.b
        return output

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        batch_size = self.input.shape[0]
        # Calculate gradients
        self.grads['W'] = np.matmul(self.input.T, grad) / batch_size
        # if self.weight_decay:
        #     self.grads['W'] += self.weight_decay_lambda * self.W
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True) / batch_size
        
        # Calculate gradients to be passed to the previous layer
        d_input = np.matmul(grad, self.W.T)
        return d_input
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}


class conv2D(Layer):
    """
    Optimized 2D convolutional layer using NumPy's vectorized operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        
        # Initialize weights and biases
        self.params = {
            'W': initialize_method(size=(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])), 
            'b': initialize_method(size=(out_channels,))
        }
        self.grads = {'W': None, 'b': None}
        
        self.input = None
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        self.padded_input = None  # Store for backward pass

    @property
    def W(self):
        return self.params['W']
    
    @W.setter
    def W(self, value):
        self.params['W'] = value
    
    @property
    def b(self):
        return self.params['b']
    
    @b.setter
    def b(self, value):
        self.params['b'] = value  

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def im2col(self, input_data, filter_h, filter_w, stride=1, pad=0):
        """
        Transform input image into column matrix for efficient convolution
        """
        N, C, H, W = input_data.shape
        out_h = (H + 2*pad - filter_h)//stride + 1
        out_w = (W + 2*pad - filter_w)//stride + 1

        img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

        for y in range(filter_h):
            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
        return col

    def col2im(self, col, input_shape, filter_h, filter_w, stride=1, pad=0):
        """
        Transform column matrix back to image format
        """
        N, C, H, W = input_shape
        out_h = (H + 2*pad - filter_h)//stride + 1
        out_w = (W + 2*pad - filter_w)//stride + 1
        col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

        img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
        for y in range(filter_h):
            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

        return img[:, :, pad:H + pad, pad:W + pad]
    
    def forward(self, X):
        """
        Vectorized forward pass using im2col
        input X: [batch, channels, H, W]
        W : [out_channels, in_channels, k, k]
        """
        self.input = X
        batch_size, _, input_height, input_width = X.shape
        
        # Apply padding and store for backward pass
        if self.padding > 0:
            self.padded_input = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), 
                                       (self.padding, self.padding)), 'constant')
        else:
            self.padded_input = X
        
        # Calculate output dimensions
        kh, kw = self.kernel_size
        output_height = (self.padded_input.shape[2] - kh) // self.stride + 1
        output_width = (self.padded_input.shape[3] - kw) // self.stride + 1
        
        # Reshape input for matrix multiplication (im2col)
        col = self.im2col(self.padded_input, kh, kw, self.stride, 0)  # padding already applied
        
        # Reshape filters for matrix multiplication
        W_col = self.W.reshape(self.out_channels, -1).T
        
        # Matrix multiplication
        out = np.matmul(col, W_col) + self.b
        
        # Reshape output
        out = out.reshape(batch_size, output_height, output_width, self.out_channels)
        out = out.transpose(0, 3, 1, 2)  # Reorder to [batch, out_channels, height, width]
        
        return out

    def backward(self, dout):
        """
        Vectorized backward pass
        dout : [batch_size, out_channel, new_H, new_W]
        """
        batch_size = self.input.shape[0]
        kh, kw = self.kernel_size
        
        # Transpose dout for easier calculations
        dout = dout.transpose(0, 2, 3, 1)  # [batch, height, width, out_channels]
        dout_reshaped = dout.reshape(-1, self.out_channels)
        
        # Calculate gradients for weights and biases
        col = self.im2col(self.padded_input, kh, kw, self.stride, 0)
        self.grads['W'] = np.matmul(col.T, dout_reshaped).T.reshape(self.W.shape)
        self.grads['b'] = np.sum(dout_reshaped, axis=0)
        
        # Calculate gradients for input
        W_col = self.W.reshape(self.out_channels, -1).T
        dcol = np.matmul(dout_reshaped, W_col.T)
        
        # Transform dcol back to image format
        dx = self.col2im(dcol, self.input.shape, kh, kw, self.stride, self.padding)
        
        # Normalize by batch size
        self.grads['W'] /= batch_size
        self.grads['b'] /= batch_size
        
        # Apply weight decay if needed
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W
        
        return dx
    
    def clear_grad(self):
        self.grads = {'W': None, 'b': None}


class PoolLayer(Layer):
    """
    Optimized pooling layer (max or average pooling).
    """
    def __init__(self, pool_type='max', kernel_size=2, stride=None, padding=0):
        super().__init__()
        self.optimizable = False
        self.pool_type = pool_type
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
            
        self.padding = padding
        self.input = None
        self.max_indices = None
        self.argmax = None  # For max pooling
    
    def __call__(self, X):
        return self.forward(X)
    
    def im2col(self, input_data, filter_h, filter_w, stride, pad):
        """
        Transform input image into column matrix for efficient pooling
        """
        N, C, H, W = input_data.shape
        out_h = (H + 2*pad - filter_h)//stride + 1
        out_w = (W + 2*pad - filter_w)//stride + 1

        img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

        for y in range(filter_h):
            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

        col = col.reshape(N, C, filter_h*filter_w, out_h, out_w)
        return col
    
    def forward(self, X):
        """
        Vectorized forward pass for pooling layer
        Input X: [batch_size, channels, height, width]
        """
        self.input = X
        N, C, H, W = X.shape
        
        kh, kw = self.kernel_size
        stride_h, stride_w = self.stride
        
        # Apply padding if needed
        if self.padding > 0:
            padded_X = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), 
                              (self.padding, self.padding)), 'constant')
        else:
            padded_X = X
        
        # Calculate output dimensions
        out_h = (padded_X.shape[2] - kh) // stride_h + 1
        out_w = (padded_X.shape[3] - kw) // stride_w + 1
        
        # Convert input to column format
        col = self.im2col(padded_X, kh, kw, stride_h, 0)  # padding already applied
        
        # Perform pooling
        if self.pool_type == 'max':
            self.argmax = np.argmax(col, axis=2)
            out = np.max(col, axis=2)
        else:  # avg pooling
            out = np.mean(col, axis=2)
        
        # Reshape output
        out = out.reshape(N, C, out_h, out_w)
        
        return out
    
    def backward(self, dout):
        """
        Vectorized backward pass for pooling layer
        """
        N, C, out_h, out_w = dout.shape
        kh, kw = self.kernel_size
        stride_h, stride_w = self.stride
        
        # Initialize gradient
        if self.padding > 0:
            dx = np.zeros((N, C, self.input.shape[2] + 2*self.padding, self.input.shape[3] + 2*self.padding))
        else:
            dx = np.zeros_like(self.input)
        
        # For average pooling, distribute gradient evenly
        if self.pool_type == 'avg':
            # Calculate dimensions
            pool_size = kh * kw
            dout_reshaped = dout.reshape(N, C, out_h, out_w, 1)
            dout_expanded = np.broadcast_to(dout_reshaped, (N, C, out_h, out_w, pool_size))
            dout_flat = dout_expanded.reshape(N, C, out_h, out_w, pool_size) / pool_size
            
            # Create a mask for average pooling (all ones)
            col = np.ones((N, C, kh*kw, out_h, out_w)) / pool_size
            
            # Calculate gradients
            for i in range(kh):
                for j in range(kw):
                    dx_h = i + np.arange(0, out_h) * stride_h
                    dx_w = j + np.arange(0, out_w) * stride_w
                    
                    # Add to dx (not using meshgrid to avoid memory issues)
                    for h_idx, h in enumerate(dx_h):
                        for w_idx, w in enumerate(dx_w):
                            dx[:, :, h, w] += dout[:, :, h_idx, w_idx] / pool_size
        
        # For max pooling, gradient flows only through max elements
        else:  # max pooling
            dout_flat = dout.reshape(N * C * out_h * out_w, 1)
            
            # Create a mask indicating max positions
            max_mask = np.zeros((N, C, kh*kw, out_h, out_w))
            
            # Get the linear index of the max values
            for n in range(N):
                for c in range(C):
                    for h in range(out_h):
                        for w in range(out_w):
                            max_mask[n, c, self.argmax[n, c, h, w], h, w] = 1
            
            # Calculate gradients
            for i in range(kh):
                for j in range(kw):
                    flat_idx = i * kw + j
                    mask = max_mask[:, :, flat_idx, :, :]
                    
                    dx_h = i + np.arange(0, out_h) * stride_h
                    dx_w = j + np.arange(0, out_w) * stride_w
                    
                    # Add to dx
                    for h_idx, h in enumerate(dx_h):
                        for w_idx, w in enumerate(dx_w):
                            dx[:, :, h, w] += dout[:, :, h_idx, w_idx] * mask[:, :, h_idx, w_idx]
        
        # Remove padding if necessary
        if self.padding > 0:
            dx = dx[:, :, self.padding:-self.padding, self.padding:-self.padding]
            
        return dx

class GlobalAvgPool(Layer):
    """
    Global Average Pooling layer that averages each feature map to a single value.
    This reduces each channel to 1x1, producing a vector of length equal to the number of channels.
    """
    def __init__(self):
        super().__init__()
        self.optimizable = False
        self.input_shape = None
    
    def __call__(self, X):
        return self.forward(X)
    
    def forward(self, X):
        """
        Input X: [batch_size, channels, height, width]
        Output: [batch_size, channels]
        """
        self.input_shape = X.shape
        return np.mean(X, axis=(2, 3))
    
    def backward(self, grads):
        """
        Input grads: [batch_size, channels]
        Output: [batch_size, channels, height, width]
        """
        batch_size, channels = grads.shape
        height, width = self.input_shape[2], self.input_shape[3]
        expanded_grads = grads.reshape(batch_size, channels, 1, 1)
        return np.broadcast_to(expanded_grads, self.input_shape) / (height * width)


class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model=None, max_classes=10) -> None:
        super().__init__()
        self.optimizable = False
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.logits = None
        self.probs = None
        self.labels = None
        self.grads = None
        self.batch_size = 0

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.logits = predicts
        self.labels = labels
        self.batch_size = predicts.shape[0]
        
        # Apply softmax if needed
        if self.has_softmax:
            self.probs = softmax(predicts)
        else:
            self.probs = predicts
        
        # Create one-hot encoded labels
        y_one_hot = np.zeros((self.batch_size, self.max_classes))
        y_one_hot[np.arange(self.batch_size), labels] = 1
        
        # Calculate cross-entropy loss
        loss = -np.sum(y_one_hot * np.log(self.probs + 1e-10)) / self.batch_size
        
        return loss
    
    def backward(self):
        # Compute gradients from the loss to the input
        y_one_hot = np.zeros((self.batch_size, self.max_classes))
        y_one_hot[np.arange(self.batch_size), self.labels] = 1
        
        if self.has_softmax:
            # If softmax is included, gradient is (softmax_output - one_hot_label)
            self.grads = (self.probs - y_one_hot) / self.batch_size
        else:
            # If softmax is not included, we need to compute gradients directly
            self.grads = -y_one_hot / (self.probs + 1e-10) / self.batch_size
        
        # Send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition

class Flatten(Layer):
    """
    Layer that flattens the input tensor from shape [batch_size, channels, height, width]
    to shape [batch_size, channels*height*width]
    """
    def __init__(self):
        super().__init__()
        self.optimizable = False
        self.input_shape = None
    
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        # Preserve batch size but flatten all other dimensions
        return X.reshape(X.shape[0], -1)
    
    def backward(self, grad):
        # Reshape gradient back to the input shape
        return grad.reshape(self.input_shape)