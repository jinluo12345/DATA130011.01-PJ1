from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers that supports pooling layers and global average pooling.
    This implementation allows customization through parameters for network architecture.
    """
    def __init__(self, conv_configs=None, fc_configs=None, act_func='ReLU', use_global_avg_pool=True):
        """
        Initialize a CNN model with customizable architecture.
        
        Args:
            conv_configs: List of dictionaries for layers. Each dict should contain:
                - type: 'conv' or 'pool'
                - For 'conv' type:
                    - in_channels: Number of input channels
                    - out_channels: Number of output channels
                    - kernel_size: Size of kernel (int or tuple)
                    - stride: Stride size (default 1)
                    - padding: Padding size (default 0)
                    - weight_decay: Whether to use weight decay (default False)
                    - weight_decay_lambda: Lambda parameter for weight decay (default 1e-8)
                - For 'pool' type:
                    - pool_type: 'max' or 'avg'
                    - kernel_size: Size of kernel (int or tuple)
                    - stride: Stride size (default equal to kernel_size)
                    - padding: Padding size (default 0)
            fc_configs: List of tuples (in_dim, out_dim) for fully connected layers
            act_func: Activation function to use ('ReLU' supported)
            use_global_avg_pool: Whether to use global average pooling before fc layers
        """
        super().__init__()
        self.layers = []
        self.optimizable = True
        self.has_initialized = False
        self.conv_configs = conv_configs
        self.fc_configs = fc_configs
        self.act_func = act_func
        self.use_global_avg_pool = use_global_avg_pool
        
        if conv_configs is not None and fc_configs is not None:
            self.has_initialized = True
            for config in conv_configs:
                layer_type = config.get('type', 'conv') 
                
                if layer_type == 'conv':
                    in_channels = config['in_channels']
                    out_channels = config['out_channels']
                    kernel_size = config['kernel_size']
                    stride = config.get('stride', 1)
                    padding = config.get('padding', 0)
                    weight_decay = config.get('weight_decay', False)
                    weight_decay_lambda = config.get('weight_decay_lambda', 1e-8)
                    
                    conv_layer = conv2D(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        weight_decay=weight_decay,
                        weight_decay_lambda=weight_decay_lambda
                    )
                    self.layers.append(conv_layer)
                    
                    if act_func == 'ReLU':
                        self.layers.append(ReLU())
                    else:
                        raise NotImplementedError(f"Activation function {act_func} not implemented")
                
                elif layer_type == 'pool':
                    pool_type = config.get('pool_type', 'max')
                    kernel_size = config['kernel_size']
                    stride = config.get('stride', kernel_size) 
                    padding = config.get('padding', 0)
                    
                    pool_layer = PoolLayer(
                        pool_type=pool_type,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding
                    )
                    self.layers.append(pool_layer)
            
            if use_global_avg_pool:
                self.layers.append(GlobalAvgPool())
            else:
                # Add a Flatten layer if not using global average pooling
                self.layers.append(Flatten())
            for i, (in_dim, out_dim) in enumerate(fc_configs):
                fc_layer = Linear(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    weight_decay=config.get('weight_decay', False),
                    weight_decay_lambda=config.get('weight_decay_lambda', 1e-8)
                )
                self.layers.append(fc_layer)
                if i < len(fc_configs) - 1 and act_func == 'ReLU':
                    self.layers.append(ReLU())

    def __call__(self, X):
        return self.forward(X)
    
    def forward(self, X):
        """
        Forward pass through the CNN.
        
        Args:
            X: Input data of shape [batch_size, channels, height, width]
            
        Returns:
            Output after forward pass through all layers
        """
        if not self.has_initialized:
            raise ValueError("Model has not been initialized. Use model.load_model to load a model "
                           "or create a new model with conv_configs and fc_configs provided.")
        
        output = X
        for layer in self.layers:
            output = layer(output)
        
        return output
    
    def backward(self, loss_grad):
        """
        Backward pass through the CNN.
        
        Args:
            loss_grad: Gradient from the loss function
            
        Returns:
            Gradient to be passed to previous layers
        """
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        
        return grad
    
    def load_model(self, param_path):
        """
        Load a pre-trained model from a file.
        
        Args:
            param_path: Path to the saved model file
        """
        with open(param_path, 'rb') as f:
            param_list = pickle.load(f)
        
        self.conv_configs = param_list[0]
        self.fc_configs = param_list[1]
        self.act_func = param_list[2]
        self.use_global_avg_pool = param_list[3]
        
        self.__init__(self.conv_configs, self.fc_configs, self.act_func, self.use_global_avg_pool)
        param_idx = 4 
        
        for i, layer in enumerate(self.layers):
            if layer.optimizable:
                layer.params['W'] = param_list[param_idx]['W']
                layer.params['b'] = param_list[param_idx]['b']
                layer.weight_decay = param_list[param_idx]['weight_decay']
                layer.weight_decay_lambda = param_list[param_idx]['lambda']
                param_idx += 1
        
        self.has_initialized = True
    
    def save_model(self, save_path):
        """
        Save the current model to a file.
        
        Args:
            save_path: Path where the model will be saved
        """
        param_list = [self.conv_configs, self.fc_configs, self.act_func, self.use_global_avg_pool]
        
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                })
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)