import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def visualize_weights(model_path, model_type='mlp', save_dir='weights_visualization'):
    """
    可视化神经网络权重
    Args:
        model_path: pickle模型文件路径
        model_type: 模型类型，'mlp' 或 'cnn'
        save_dir: 结果保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载模型参数
    try:
        with open(model_path, 'rb') as f:
            params = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 根据模型类型处理
    if model_type.lower() == 'mlp':
        visualize_mlp_weights(params, save_dir)
    elif model_type.lower() == 'cnn':
        visualize_cnn_weights(params, save_dir)
    else:
        print(f"Unsupported model type: {model_type}")

def visualize_mlp_weights(params, save_dir):
    """可视化MLP第一层权重"""
    try:
        # 解析MLP参数结构
        size_list = params[0]
        act_func = params[1]
        first_layer_params = params[2]  # 第一个全连接层参数
        
        # 获取权重矩阵 (in_dim, out_dim)
        W = first_layer_params['W']
        
        # 转置为(out_dim, in_dim)以便按输出神经元选择
        W = W.T
        
        # 取前9个输出神经元的权重
        n_show = min(9, W.shape[0])
        weights = W[:n_show]
        
        # 创建子图
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        fig.suptitle('MLP First Layer Weights', fontsize=16)
        
        # 可视化每个权重
        for i, ax in enumerate(axes.flat):
            if i < n_show:
                # 转换为28x28图像
                img = weights[i].reshape(28, 28)
                
                # 归一化到0-1
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                
                ax.imshow(img, cmap='gray')
                ax.set_title(f'Neuron {i+1}')
                ax.axis('off')
            else:
                ax.axis('off')
        
        # 保存并显示
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'mlp_first_layer_weights.png')
        plt.savefig(save_path, dpi=150)
        print(f"MLP weights saved to {save_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error visualizing MLP weights: {e}")

def visualize_cnn_weights(params, save_dir):
    """可视化CNN第一层卷积核"""
    try:
        # 解析CNN参数结构
        conv_configs = params[0]
        first_conv_params = params[4]  # 第一个卷积层参数
        
        # 获取卷积核权重 (out_channels, in_channels, kH, kW)
        W = first_conv_params['W']
        
        # 取前16个卷积核（假设输入通道为1）
        n_show = min(16, W.shape[0])
        kernels = W[:n_show, 0]  # 取第一个输入通道的核
        
        # 创建子图
        grid_size = int(np.ceil(np.sqrt(n_show)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        fig.suptitle('CNN First Layer Kernels', fontsize=16)
        
        # 可视化每个卷积核
        for i, ax in enumerate(axes.flat):
            if i < n_show:
                kernel = kernels[i]
                
                # 归一化到0-1
                kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-8)
                
                ax.imshow(kernel, cmap='gray')
                ax.set_title(f'Kernel {i+1}')
                ax.axis('off')
            else:
                ax.axis('off')
        
        # 保存并显示
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'cnn_first_layer_kernels.png')
        plt.savefig(save_path, dpi=150)
        print(f"CNN kernels saved to {save_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error visualizing CNN weights: {e}")

if __name__ == "__main__":
    # 使用示例
    mlp_model_path = r'D:\learning\fudan\神经网络与深度学习\PJ1\PJ1\codes\ablation_results\MLP_20250418_120146\custom_mlp\best_model.pickle' 
    cnn_model_path = r'D:\learning\fudan\神经网络与深度学习\PJ1\PJ1\codes\ablation_results\CNN_20250418_024613\cnn_baseline\best_model.pickle' 
    visualize_weights(mlp_model_path, model_type='mlp')
    visualize_weights(cnn_model_path, model_type='cnn')