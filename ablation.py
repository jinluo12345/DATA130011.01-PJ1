import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
import time
from datetime import datetime
import json
import argparse
import itertools
from scipy.ndimage import rotate, shift


class AblationStudy:
    """
    A class for performing ablation studies on MLP and CNN models.
    """
    def __init__(self, model_type="MLP", base_dir="ablation_results"):
        """
        Initialize the ablation study.
        
        Args:
            model_type (str): Type of model to use ('MLP' or 'CNN')
            base_dir (str): Base directory to save results
        """
        self.model_type = model_type
        self.base_dir = base_dir
        
        # Create base directory if it doesn't exist
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            
        # Create a timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_dir, f"{model_type}_{self.timestamp}")
        os.makedirs(self.run_dir)
        
        # Load the dataset
        self.train_imgs, self.train_labs, self.valid_imgs, self.valid_labs, self.test_imgs, self.test_labs = self._load_data()
        
        # Store results
        self.results = []

    def _load_data(self):
        """
        Load and preprocess the MNIST dataset.
        
        Returns:
            tuple: Train, validation, and test datasets
        """
        print("Loading dataset...")
        
        # Set paths
        train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
        train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'
        test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
        test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'
        
        # Load training data
        with gzip.open(train_images_path, 'rb') as f:
            magic, num, rows, cols = unpack('>4I', f.read(16))
            train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
        
        with gzip.open(train_labels_path, 'rb') as f:
            magic, num = unpack('>2I', f.read(8))
            train_labs = np.frombuffer(f.read(), dtype=np.uint8)
        
        # Load test data
        with gzip.open(test_images_path, 'rb') as f:
            magic, num, rows, cols = unpack('>4I', f.read(16))
            test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
        
        with gzip.open(test_labels_path, 'rb') as f:
            magic, num = unpack('>2I', f.read(8))
            test_labs = np.frombuffer(f.read(), dtype=np.uint8)
        
        # Create validation set
        np.random.seed(309)
        idx = np.random.permutation(np.arange(train_imgs.shape[0]))
        train_imgs = train_imgs[idx]
        train_labs = train_labs[idx]
        valid_imgs = train_imgs[:10000]
        valid_labs = train_labs[:10000]
        train_imgs = train_imgs[10000:]
        train_labs = train_labs[10000:]
        
        # Normalize data
        train_imgs = train_imgs / train_imgs.max()
        valid_imgs = valid_imgs / valid_imgs.max()
        test_imgs = test_imgs / test_imgs.max()
        
        # For CNN, reshape the data
        if self.model_type == "CNN":
            train_imgs = train_imgs.reshape(-1, 1, 28, 28)
            valid_imgs = valid_imgs.reshape(-1, 1, 28, 28)
            test_imgs = test_imgs.reshape(-1, 1, 28, 28)
        
        return train_imgs, train_labs, valid_imgs, valid_labs, test_imgs, test_labs

    def create_mlp_model(self, hidden_layers, activation='ReLU', weight_decay=None):
        """
        Create an MLP model.
        
        Args:
            hidden_layers (list): List of hidden layer sizes
            activation (str): Activation function to use
            weight_decay (list): Weight decay parameters for each layer
            
        Returns:
            nn.models.Model_MLP: An MLP model
        """
        # Prepare layer sizes
        size_list = [self.train_imgs.shape[-1]] + hidden_layers + [10]
        
        # Set weight decay
        if weight_decay is None:
            weight_decay = [1e-4] * (len(size_list) - 1)
        
        return nn.models.Model_MLP(size_list, activation, weight_decay)

    def create_cnn_model(self, conv_configs, fc_configs, activation='ReLU', use_global_avg_pool=True):
        """
        Create a CNN model.
        
        Args:
            conv_configs (list): List of convolutional layer configurations
            fc_configs (list): List of fully connected layer configurations
            activation (str): Activation function to use
            use_global_avg_pool (bool): Whether to use global average pooling
            
        Returns:
            nn.models.Model_CNN: A CNN model
        """
        return nn.models.Model_CNN(conv_configs, fc_configs, activation, use_global_avg_pool)

    def create_optimizer(self, optimizer_name, init_lr, model, mu=0.9):
        """
        Create an optimizer.
        
        Args:
            optimizer_name (str): Name of the optimizer to use
            init_lr (float): Initial learning rate
            model: Model to optimize
            mu (float): Momentum parameter for MomentGD
            
        Returns:
            nn.optimizer.Optimizer: An optimizer
        """
        if optimizer_name == "SGD":
            return nn.optimizer.SGD(init_lr=init_lr, model=model)
        elif optimizer_name == "MomentGD":
            return nn.optimizer.MomentGD(init_lr=init_lr, model=model, mu=mu)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def create_scheduler(self, scheduler_name, optimizer, **kwargs):
        """
        Create a learning rate scheduler.
        
        Args:
            scheduler_name (str): Name of the scheduler to use
            optimizer: Optimizer to schedule
            **kwargs: Additional parameters for the scheduler
            
        Returns:
            nn.lr_scheduler: A learning rate scheduler
        """
        if scheduler_name == "StepLR":
            step_size = kwargs.get("step_size", 30)
            gamma = kwargs.get("gamma", 0.1)
            return nn.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == "MultiStepLR":
            milestones = kwargs.get("milestones", [30, 60, 90])
            gamma = kwargs.get("gamma", 0.1)
            return nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=gamma)
        elif scheduler_name == "ExponentialLR":
            gamma = kwargs.get("gamma", 0.95)
            return nn.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    def run_experiment(self, config):
        """
        Run a single experiment with the given configuration.
        
        Args:
            config (dict): Configuration for the experiment
            
        Returns:
            dict: Results of the experiment
        """
        experiment_name = config.get("name", f"experiment_{len(self.results)}")
        print(f"\n\n{'='*50}")
        print(f"Running experiment: {experiment_name}")
        print(f"{'='*50}\n")
        
        # Create experiment directory
        experiment_dir = os.path.join(self.run_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Save configuration
        with open(os.path.join(experiment_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
        
        # Create model
        if self.model_type == "MLP":
            model = self.create_mlp_model(
                hidden_layers=config.get("hidden_layers", [128]),
                activation=config.get("activation", "ReLU"),
                weight_decay=config.get("weight_decay", None)
            )
        else:  # CNN
            model = self.create_cnn_model(
                conv_configs=config.get("conv_configs"),
                fc_configs=config.get("fc_configs"),
                activation=config.get("activation", "ReLU"),
                use_global_avg_pool=config.get("use_global_avg_pool", True)
            )
        
        # Create optimizer
        optimizer = self.create_optimizer(
            optimizer_name=config.get("optimizer", "SGD"),
            init_lr=config.get("learning_rate", 0.01),
            model=model,
            mu=config.get("momentum", 0.9)
        )
        
        # Create scheduler
        scheduler = self.create_scheduler(
            scheduler_name=config.get("scheduler", "MultiStepLR"),
            optimizer=optimizer,
            **config.get("scheduler_params", {})
        )
        
        # Create loss function
        loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)

        # Create runner
        augmentation_config = config.get("augmentation", {"enabled": False})
        runner = nn.runner.RunnerM(
            model=model,
            optimizer=optimizer,
            metric=nn.metric.accuracy,
            loss_fn=loss_fn,
            batch_size=config.get("batch_size", 32),
            scheduler=scheduler,
            augmentation_config=augmentation_config
        )
        
        # Start timing
        start_time = time.time()
        
        # Train model
        runner.train(
            train_set=[self.train_imgs, self.train_labs],
            dev_set=[self.valid_imgs, self.valid_labs],
            num_epochs=config.get("num_epochs", 5),
            log_iters=config.get("log_iters", 100),
            save_dir=experiment_dir
        )
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Store the runner temporarily to access the learning curves
        self._temp_runner = runner
        
        # Save runner data for later visualization
        with open(os.path.join(experiment_dir, "runner_data.pkl"), "wb") as f:
            pickle.dump({
                "train_scores": runner.train_scores,
                "dev_scores": runner.dev_scores,
                "train_loss": runner.train_loss,
                "dev_loss": runner.dev_loss,
                "dev_steps": runner.dev_steps  
            }, f)
        
        # Evaluate on test set
        test_accuracy, test_loss = runner.evaluate([self.test_imgs, self.test_labs])
        
        # Save learning curves
        self._save_learning_curves(
            runner=runner,
            save_path=os.path.join(experiment_dir, "learning_curves.png"),
            title=f"{experiment_name} Learning Curves"
        )
        
        # Prepare results
        result = {
            "name": experiment_name,
            "config": config,
            "training_time": training_time,
            "best_validation_accuracy": runner.best_score,
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
            "final_train_accuracy": runner.train_scores[-1],
            "final_train_loss": runner.train_loss[-1]
        }
        
        # Save result
        self.results.append(result)
        
        # Print summary
        print(f"\nExperiment '{experiment_name}' completed:")
        print(f"  Training time: {training_time:.2f} seconds")
        print(f"  Best validation accuracy: {runner.best_score:.4f}")
        print(f"  Test accuracy: {test_accuracy:.4f}")
        print(f"  Test loss: {test_loss:.4f}")
        
        return result

    def _save_learning_curves(self, runner, save_path, title="Learning Curves"):
        """
        Save learning curves as a figure with horizontal subplots.
        
        Args:
            runner: Runner with training history
            save_path (str): Path to save the figure
            title (str): Title for the figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))  # Changed from (2, 1) to (1, 2)
        fig.suptitle(title, fontsize=16)
        
        # Plot accuracy
        axes[0].plot(range(len(runner.train_scores)), runner.train_scores, label="Train")
        axes[0].plot(runner.dev_steps, runner.dev_scores, label="Validation")
        axes[0].set_title("Accuracy")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot loss
        axes[1].plot(range(len(runner.train_scores)),runner.train_loss, label="Train")
        axes[1].plot(runner.dev_steps,runner.dev_loss, label="Validation")
        axes[1].set_title("Loss")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(True)
        
        fig.tight_layout(rect=[0, 0, 1, 0.94])  # Adjusted for the suptitle
        fig.savefig(save_path)
        plt.close(fig)

    def save_results(self):
        """
        Save all results to a CSV file and generate a summary figure.
        """
        # Create DataFrame from results
        df = pd.DataFrame(self.results)
        
        # Save to CSV
        csv_path = os.path.join(self.run_dir, "results.csv")
        df.to_csv(csv_path, index=False)
        
        # Save summary table as HTML
        html_table = df[["name", "training_time", "best_validation_accuracy", "test_accuracy", "test_loss"]].to_html()
        with open(os.path.join(self.run_dir, "summary_table.html"), "w") as f:
            f.write(html_table)
        
        # Generate summary figure
        self._generate_summary_figure()
        
        # Generate combined learning curves from saved runner data
        self._load_and_plot_all_learning_curves()
        
        print(f"\nResults saved to {self.run_dir}")
        print(f"Summary table: {os.path.join(self.run_dir, 'summary_table.html')}")
        print(f"Full results: {csv_path}")
        print(f"Combined learning curves: {os.path.join(self.run_dir, 'combined_learning_curves.png')}")
        
    def _load_and_plot_all_learning_curves(self, smoothing_factor=0.8):
        """
        Load saved runner data for all experiments and plot combined learning curves.
        
        Args:
            smoothing_factor (float): Factor for exponential moving average smoothing
        """
        if not self.results:
            return
            
        # Create figure with two subplots (accuracy and loss) side by side
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))  # Changed from (2, 1) to (1, 2)
        fig.suptitle("Combined Learning Curves (Smoothed)", fontsize=16)
        
        # Define a colormap for the different experiments
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.results)))
        
        # Plot each experiment's learning curves
        for i, result in enumerate(self.results):
            experiment_name = result["name"]
            experiment_dir = os.path.join(self.run_dir, experiment_name)
            runner_data_path = os.path.join(experiment_dir, "runner_data.pkl")
            
            try:
                # Load the saved runner data
                with open(runner_data_path, "rb") as f:
                    runner_data = pickle.load(f)
                    
                train_scores = runner_data["train_scores"]
                dev_scores = runner_data["dev_scores"]
                train_loss = runner_data["train_loss"]
                dev_loss = runner_data["dev_loss"]
                dev_steps = runner_data["dev_steps"]

                # Apply smoothing
                smoothed_train_acc = self._smooth_curve(train_scores, smoothing_factor)
                smoothed_val_acc = self._smooth_curve(dev_scores, smoothing_factor)
                smoothed_train_loss = self._smooth_curve(train_loss, smoothing_factor)
                smoothed_val_loss = self._smooth_curve(dev_loss, smoothing_factor)
                
                # Plot accuracy
                axes[0].plot(smoothed_train_acc, color=colors[i], linestyle='-', alpha=0.7,
                            label=f"{experiment_name} (Train)")
                axes[0].plot(dev_steps, smoothed_val_acc, color=colors[i], linestyle='--', label=f"{experiment_name} (Val)")
        
                
                # Plot loss
                axes[1].plot(smoothed_train_loss, color=colors[i], linestyle='-', alpha=0.7,
                            label=f"{experiment_name} (Train)")
                axes[1].plot(dev_steps, smoothed_val_loss, color=colors[i], linestyle='--', label=f"{experiment_name} (Val)")
                            
                print(f"Added learning curves for experiment: {experiment_name}")
                
            except (FileNotFoundError, KeyError) as e:
                print(f"Warning: Could not load learning curves for {experiment_name}: {e}")
                continue
        
        # Setup accuracy subplot
        axes[0].set_title("Accuracy")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend(loc='lower right')
        axes[0].grid(True)
        
        # Setup loss subplot
        axes[1].set_title("Loss")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Loss")
        axes[1].legend(loc='upper right')
        axes[1].grid(True)
        
        fig.tight_layout(rect=[0, 0, 1, 0.94])  # Adjusted for the suptitle
        fig.savefig(os.path.join(self.run_dir, "combined_learning_curves.png"))
        
        # Create a separate figure for just the validation curves (often cleaner)
        fig_val, axes_val = plt.subplots(1, 2, figsize=(20, 10))  # Changed from (2, 1) to (1, 2)
        fig_val.suptitle("Validation Curves Comparison (Smoothed)", fontsize=16)
        
        for i, result in enumerate(self.results):
            experiment_name = result["name"]
            experiment_dir = os.path.join(self.run_dir, experiment_name)
            runner_data_path = os.path.join(experiment_dir, "runner_data.pkl")
            
            try:
                # Load the saved runner data
                with open(runner_data_path, "rb") as f:
                    runner_data = pickle.load(f)
                    
                dev_scores = runner_data["dev_scores"]
                dev_loss = runner_data["dev_loss"]
                dev_steps = runner_data["dev_steps"]
                # Apply smoothing
                smoothed_val_acc = self._smooth_curve(dev_scores, smoothing_factor)
                smoothed_val_loss = self._smooth_curve(dev_loss, smoothing_factor)
                
                # Plot only validation accuracy and loss
                axes_val[0].plot(dev_steps,smoothed_val_acc, color=colors[i], linestyle='-',
                            label=f"{experiment_name}")
                axes_val[1].plot(dev_steps,smoothed_val_loss, color=colors[i], linestyle='-',
                            label=f"{experiment_name}")
                
            except (FileNotFoundError, KeyError):
                continue
        
        # Setup validation accuracy subplot
        axes_val[0].set_title("Validation Accuracy")
        axes_val[0].set_xlabel("Iteration")
        axes_val[0].set_ylabel("Accuracy")
        axes_val[0].legend(loc='lower right')
        axes_val[0].grid(True)
        
        # Setup validation loss subplot
        axes_val[1].set_title("Validation Loss")
        axes_val[1].set_xlabel("Iteration")
        axes_val[1].set_ylabel("Loss")
        axes_val[1].legend(loc='upper right')
        axes_val[1].grid(True)
        
        fig_val.tight_layout(rect=[0, 0, 1, 0.94])  # Adjusted for the suptitle
        fig_val.savefig(os.path.join(self.run_dir, "validation_curves_comparison.png"))
        
        plt.close(fig)
        plt.close(fig_val)

    def _generate_summary_figure(self):
        """
        Generate a summary figure comparing test accuracies of all experiments.
        """
        if not self.results:
            return
        
        names = [result["name"] for result in self.results]
        test_accuracies = [result["test_accuracy"] for result in self.results]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(names, test_accuracies)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', rotation=0)
        
        plt.title("Test Accuracy Comparison")
        plt.xlabel("Experiment")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, "accuracy_comparison.png"))
        plt.close()
        
        # Create training time comparison
        training_times = [result["training_time"] for result in self.results]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(names, training_times)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s',
                    ha='center', va='bottom', rotation=0)
        
        plt.title("Training Time Comparison")
        plt.xlabel("Experiment")
        plt.ylabel("Time (seconds)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, "time_comparison.png"))
        plt.close()
        
        # Generate smoothed learning curves for all experiments
        self._generate_combined_learning_curves()
    
    def _smooth_curve(self, points, factor=0.8):
        """
        Apply exponential moving average to smooth the curve.
        
        Args:
            points (array): Data points to smooth
            factor (float): Smoothing factor between 0 and 1
                            (higher means more smoothing)
                            
        Returns:
            array: Smoothed data points
        """
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points
    
    def _generate_combined_learning_curves(self):
        """
        Generate combined learning curves for all experiments with smoothing.
        """
        if not self.results:
            return
            
        # Create figure with two subplots (accuracy and loss) side by side
        fig, axes = plt.subplots(1, 2, figsize=(18, 10))  # Changed from (2, 1) to (1, 2)
        fig.suptitle("Combined Learning Curves (Smoothed)", fontsize=16)
        
        # Define a colormap for the different experiments
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.results)))
        
        # Plot smoothed accuracy curves
        for i, result in enumerate(self.results):
            # Get runner data from saved experiment directory
            experiment_dir = os.path.join(self.run_dir, result["name"])
            
            # If we have direct access to the runner data (in memory)
            # Note: In real-world usage, this data might need to be loaded from saved files
            runner_data_path = os.path.join(experiment_dir, "runner_data.pkl")
            
            # Try to access run_experiment's runner directly
            if hasattr(self, '_temp_runner'):
                runner = self._temp_runner
                train_scores = runner.train_scores
                dev_scores = runner.dev_scores
                train_loss = runner.train_loss
                dev_loss = runner.dev_loss
            else:
                # Use the data from the result
                # We'll use iteration as x-axis (simply the indices)
                iterations = range(len(result['config']['num_epochs'] * len(self.train_imgs) // result['config']['batch_size']))
                
                # Since we might not have direct access to all curves, we'll plot what we can
                train_acc = result.get("final_train_accuracy")
                val_acc = result.get("best_validation_accuracy")
                test_acc = result.get("test_accuracy")
                
                # Print information about what we're plotting
                print(f"Experiment {result['name']}:")
                print(f"  Final Train Accuracy: {train_acc:.4f}")
                print(f"  Best Validation Accuracy: {val_acc:.4f}")
                print(f"  Test Accuracy: {test_acc:.4f}")
                
                # Skip plotting detailed curves if we don't have the data
                continue
            
            # Apply smoothing
            smoothed_train_acc = self._smooth_curve(train_scores)
            smoothed_val_acc = self._smooth_curve(dev_scores)
            smoothed_train_loss = self._smooth_curve(train_loss)
            smoothed_val_loss = self._smooth_curve(dev_loss)
            
            # Plot accuracy
            axes[0].plot(smoothed_train_acc, color=colors[i], linestyle='-', alpha=0.7,
                        label=f"{result['name']} (Train)")
            axes[0].plot(smoothed_val_acc, color=colors[i], linestyle='--',
                        label=f"{result['name']} (Val)")
            
            # Plot loss
            axes[1].plot(smoothed_train_loss, color=colors[i], linestyle='-', alpha=0.7,
                        label=f"{result['name']} (Train)")
            axes[1].plot(smoothed_val_loss, color=colors[i], linestyle='--',
                        label=f"{result['name']} (Val)")
        
        # Setup accuracy subplot
        axes[0].set_title("Accuracy")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend(loc='lower right')
        axes[0].grid(True)
        
        # Setup loss subplot
        axes[1].set_title("Loss")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Loss")
        axes[1].legend(loc='upper right')
        axes[1].grid(True)
        
        fig.tight_layout(rect=[0, 0, 1, 0.94])  # Adjusted for the suptitle
        fig.savefig(os.path.join(self.run_dir, "combined_learning_curves.png"))
        plt.close(fig)
    def run_mlp_ablation(self, configs=None):
        """
        Run an ablation study for MLP models.
        
        Args:
            configs (list): List of configurations to run
        """
        if configs is None:
            # Default configurations for ablation study
            configs = [
                {
                    "name": "mlp_baseline",
                    "hidden_layers": [128],
                    "activation": "ReLU",
                    "optimizer": "SGD",
                    "learning_rate": 0.1,
                    "scheduler": "ExponentialLR",
                    "scheduler_params": {"gamma": 1.0},
                    "batch_size": 32,
                    "num_epochs": 10,
                    "log_iters": 100
                },
                {
                    "name": "mlp_wd_1e-4",
                    "hidden_layers": [128],
                    "activation": "ReLU",
                    "weight_decay": [1e-4, 1e-4], 
                    "optimizer": "SGD",
                    "learning_rate": 0.1,
                    "scheduler": "ExponentialLR",
                    "scheduler_params": {"gamma": 1.0},
                    "batch_size": 32,
                    "num_epochs": 10,
                    "log_iters": 100
                },
                {
                    "name": "mlp_wd_1e-2",
                    "hidden_layers": [128],
                    "activation": "ReLU",
                    "weight_decay": [1e-2, 1e-2],  
                    "optimizer": "SGD",
                    "learning_rate": 0.1,
                    "scheduler": "ExponentialLR", 
                    "scheduler_params": {"gamma": 1.0},
                    "batch_size": 32,
                    "num_epochs": 10,
                    "log_iters": 100
                },
                # {
                #     "name": "mlp_deep",
                #     "hidden_layers": [128, 64],
                #     "activation": "ReLU",
                #     "optimizer": "SGD",
                #     "learning_rate": 0.1,
                #     "scheduler": "ExponentialLR",
                #     "scheduler_params": {"gamma": 1.0},
                #     "batch_size": 32,
                #     "num_epochs": 5,
                #     "log_iters": 100
                # },
                # {
                #     "name": "mlp_wide",
                #     "hidden_layers": [512],
                #     "activation": "ReLU",
                #     "optimizer": "SGD",
                #     "learning_rate": 0.1,
                #     "scheduler": "ExponentialLR",
                #     "scheduler_params": {"gamma": 1.0},
                #     "batch_size": 32,
                #     "num_epochs": 5,
                #     "log_iters": 100
                # },
                # {
                #     "name": "mlp_wide_deep",
                #     "hidden_layers": [512,128],
                #     "activation": "ReLU",
                #     "optimizer": "SGD",
                #     "learning_rate": 0.1,
                #     "scheduler": "ExponentialLR",
                #     "scheduler_params": {"gamma": 1.0},
                #     "batch_size": 32,
                #     "num_epochs": 5,
                #     "log_iters": 100
                # },
                # {
                #     "name": "mlp_with_shift",
                #     "hidden_layers": [128],
                #     "activation": "ReLU",
                #     "optimizer": "SGD",
                #     "learning_rate": 0.1,
                #     "scheduler": "ExponentialLR",
                #     "scheduler_params": {"gamma": 1.0},
                #     "batch_size": 32,
                #     "num_epochs": 100,
                #     "log_iters": 100,
                #     "augmentation": {
                #         "enabled": True,
                #         "rotation_prob": 0.0,
                #         "shift_prob": 0.5,
                #         "noise_prob": 0.0,
                #         "rotation_range": (-25, 25),
                #         "shift_range": (-4, 4),
                #         "noise_level": 0.08
                #     }
                # },
                # {
                #     "name": "mlp_with_rotate",
                #     "hidden_layers": [128],
                #     "activation": "ReLU",
                #     "optimizer": "SGD",
                #     "learning_rate": 0.1,
                #     "scheduler": "ExponentialLR",
                #     "scheduler_params": {"gamma": 1.0},
                #     "batch_size": 32,
                #     "num_epochs": 100,
                #     "log_iters": 100,
                #     "augmentation": {
                #         "enabled": True,
                #         "rotation_prob": 0.5,
                #         "shift_prob": 0.0,
                #         "noise_prob": 0.0,
                #         "rotation_range": (-25, 25),
                #         "shift_range": (-4, 4),
                #         "noise_level": 0.08
                #     }
                # },
                # {
                #     "name": "mlp_with_noise",
                #     "hidden_layers": [128],
                #     "activation": "ReLU",
                #     "optimizer": "SGD",
                #     "learning_rate": 0.1,
                #     "scheduler": "ExponentialLR",
                #     "scheduler_params": {"gamma": 1.0},
                #     "batch_size": 32,
                #     "num_epochs": 100,
                #     "log_iters": 100,
                #     "augmentation": {
                #         "enabled": True,
                #         "rotation_prob": 0.0,
                #         "shift_prob": 0.0,
                #         "noise_prob": 0.5,
                #         "rotation_range": (-25, 25),
                #         "shift_range": (-4, 4),
                #         "noise_level": 0.08
                #     }
                # },
                # {
                #     "name": "mlp_with_all_aug",
                #     "hidden_layers": [128],
                #     "activation": "ReLU",
                #     "optimizer": "SGD",
                #     "learning_rate": 0.1,
                #     "scheduler": "ExponentialLR",
                #     "scheduler_params": {"gamma": 1.0},
                #     "batch_size": 32,
                #     "num_epochs": 100,
                #     "log_iters": 100,
                #     "augmentation": {
                #         "enabled": True,
                #         "rotation_prob": 0.3,
                #         "shift_prob": 0.3,
                #         "noise_prob": 0.3,
                #         "rotation_range": (-25, 25),
                #         "shift_range": (-4, 4),
                #         "noise_level": 0.08
                #     }
                # },
                # {
                #     "name": "mlp_momentum",
                #     "hidden_layers": [128],
                #     "activation": "ReLU",
                #     "optimizer": "MomentGD",
                #     "learning_rate": 0.1,
                #     "momentum": 0.9,
                #     "scheduler": "ExponentialLR",
                #     "scheduler_params": {"gamma": 1.0},
                #     "batch_size": 32,
                #     "num_epochs": 5,
                #     "log_iters": 100
                # },
                # {
                #     "name": "mlp_heavy_momentum",
                #     "hidden_layers": [128],
                #     "activation": "ReLU",
                #     "optimizer": "MomentGD",
                #     "learning_rate": 0.1,
                #     "momentum": 0.999,
                #     "scheduler": "ExponentialLR",
                #     "scheduler_params": {"gamma": 1.0},
                #     "batch_size": 32,
                #     "num_epochs": 5,
                #     "log_iters": 100
                # },

                # {
                #     "name": "mlp_step_lr",
                #     "hidden_layers": [128],
                #     "activation": "ReLU",
                #     "optimizer": "SGD",
                #     "learning_rate": 0.01,
                #     "scheduler": "StepLR",
                #     "scheduler_params": {"step_size": 30, "gamma": 0.5},
                #     "batch_size": 32,
                #     "num_epochs": 5,
                #     "log_iters": 100
                # },
                # {
                #     "name": "mlp_exp_lr",
                #     "hidden_layers": [128],
                #     "activation": "ReLU",
                #     "optimizer": "SGD",
                #     "learning_rate": 0.01,
                #     "scheduler": "ExponentialLR",
                #     "scheduler_params": {"milestones": [30, 60, 90], "gamma": 0.1},
                #     "batch_size": 32,
                #     "num_epochs": 5,
                #     "log_iters": 100
                # }
            ]
        
        # Run each configuration
        for config in configs:
            self.run_experiment(config)
        
        # Save results
        self.save_results()

    def run_cnn_ablation(self, configs=None):
        """
        Run an ablation study for CNN models.
        
        Args:
            configs (list): List of configurations to run
        """
        if configs is None:
            # Default configurations for CNN ablation study
            configs = [
                {
                    "name": "cnn_baseline",
                    "conv_configs": [
                        {"type": "conv", "in_channels": 1, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
                        {"type": "pool", "pool_type": "max", "kernel_size": 2}
                    ],
                    "fc_configs": [(16 * 14 * 14, 64),(64,10)],
                    "activation": "ReLU",
                    "use_global_avg_pool": False,
                    "optimizer": "SGD",
                    "learning_rate": 0.1,
                    "scheduler": "ExponentialLR",
                    "scheduler_params": {"gamma": 1.0},
                    "batch_size": 32,
                    "num_epochs": 5,
                    "log_iters": 100
                },
                {
                    "name": "cnn_deep",
                    "conv_configs": [
                        {"type": "conv", "in_channels": 1, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
                        {"type": "pool", "pool_type": "max", "kernel_size": 2},
                        {"type": "conv", "in_channels": 16, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
                        {"type": "pool", "pool_type": "max", "kernel_size": 2},
                        {"type": "conv", "in_channels": 32, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1}
                    ],
                    "fc_configs": [(64 * 7 * 7, 10)],
                    "activation": "ReLU",
                    "use_global_avg_pool": False,
                    "optimizer": "SGD",
                    "learning_rate": 0.1,
                    "scheduler": "ExponentialLR",
                    "scheduler_params": {"gamma": 1.0},
                    "batch_size": 32,
                    "num_epochs": 5,
                    "log_iters": 100
                },
                {
                    "name": "cnn_large",
                    "conv_configs": [
                        {"type": "conv", "in_channels": 1, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
                        {"type": "pool", "pool_type": "max", "kernel_size": 2},
                    ],
                    "fc_configs": [(64 * 14 * 14,10)],
                    "activation": "ReLU",
                    "use_global_avg_pool": False,
                    "optimizer": "SGD",
                    "learning_rate": 0.1,
                    "scheduler": "ExponentialLR",
                    "scheduler_params": {"gamma": 1.0},
                    "batch_size": 32,
                    "num_epochs": 5,
                    "log_iters": 100
                },

                # {
                #     "name": "cnn_global_avg_pool",
                #     "conv_configs": [
                #         {"type": "conv", "in_channels": 1, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
                #         {"type": "pool", "pool_type": "max", "kernel_size": 2},
                #         {"type": "conv", "in_channels": 16, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
                #         {"type": "pool", "pool_type": "max", "kernel_size": 2},
                #         {"type": "conv", "in_channels": 32, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1}
                #     ],
                #     "fc_configs": [(64, 10)],
                #     "activation": "ReLU",
                #     "use_global_avg_pool": True,
                #     "optimizer": "SGD",
                #     "learning_rate": 0.01,
                #     "scheduler": "MultiStepLR",
                #     "scheduler_params": {"milestones": [30, 60, 90], "gamma": 0.1},
                #     "batch_size": 32,
                #     "num_epochs": 5,
                #     "log_iters": 100
                # },
                # {
                #     "name": "cnn_momentum",
                #     "conv_configs": [
                #         {"type": "conv", "in_channels": 1, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
                #         {"type": "pool", "pool_type": "max", "kernel_size": 2},
                #         {"type": "conv", "in_channels": 16, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
                #         {"type": "pool", "pool_type": "max", "kernel_size": 2}
                #     ],
                #     "fc_configs": [(32 * 7 * 7, 128), (128, 10)],
                #     "activation": "ReLU",
                #     "use_global_avg_pool": False,
                #     "optimizer": "MomentGD",
                #     "learning_rate": 0.01,
                #     "momentum": 0.9,
                #     "scheduler": "MultiStepLR",
                #     "scheduler_params": {"milestones": [30, 60, 90], "gamma": 0.1},
                #     "batch_size": 32,
                #     "num_epochs": 5,
                #     "log_iters": 100
                # },
                # {
                #     "name": "cnn_exp_lr",
                #     "conv_configs": [
                #         {"type": "conv", "in_channels": 1, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
                #         {"type": "pool", "pool_type": "max", "kernel_size": 2},
                #         {"type": "conv", "in_channels": 16, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
                #         {"type": "pool", "pool_type": "max", "kernel_size": 2}
                #     ],
                #     "fc_configs": [(32 * 7 * 7, 128), (128, 10)],
                #     "activation": "ReLU",
                #     "use_global_avg_pool": False,
                #     "optimizer": "SGD",
                #     "learning_rate": 0.01,
                #     "scheduler": "ExponentialLR",
                #     "scheduler_params": {"gamma": 0.95},
                #     "batch_size": 32,
                #     "num_epochs": 5,
                #     "log_iters": 100
                # }
            ]
        
        # Run each configuration
        for config in configs:
            self.run_experiment(config)
        
        # Save results
        self.save_results()

    def generate_grid_search_configs(self, param_grid):
        """
        Generate configurations for a grid search.
        
        Args:
            param_grid (dict): Parameters and their possible values
            
        Returns:
            list: List of configuration dictionaries
        """
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        configs = []
        
        # Generate all combinations of parameters
        for i, values in enumerate(itertools.product(*param_values)):
            config = {param_names[j]: values[j] for j in range(len(param_names))}
            config["name"] = f"grid_search_{i+1}"
            configs.append(config)
        
        return configs
        

def main():
    """
    Main function to run the ablation study.
    """
    parser = argparse.ArgumentParser(description="Run ablation studies for neural network models.")
    parser.add_argument("--model", type=str, choices=["MLP", "CNN"], default="CNN", help="Type of model to use")
    parser.add_argument("--mode", type=str, choices=["ablation", "grid_search", "single"], default="ablation", help="Mode to run")
    parser.add_argument("--output_dir", type=str, default="ablation_results", help="Directory to save results")
    args = parser.parse_args()
    
    # Create ablation study object
    study = AblationStudy(model_type=args.model, base_dir=args.output_dir)
    
    if args.mode == "ablation":
        # Run standard ablation
        if args.model == "MLP":
            study.run_mlp_ablation()
        else:  # CNN
            study.run_cnn_ablation()
    elif args.mode == "grid_search":
        # Example grid search for MLP
        if args.model == "MLP":
            param_grid = {
                "hidden_layers": [[64], [128], [256], [128, 64]],
                "learning_rate": [0.001, 0.01, 0.1],
                "optimizer": ["SGD", "MomentGD"]
            }
            configs = study.generate_grid_search_configs(param_grid)
            for config in configs:
                study.run_experiment(config)
            study.save_results()
        else:  # CNN
            # Example grid search for CNN
            base_conv_config = [
                {"type": "conv", "in_channels": 1, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
                {"type": "pool", "pool_type": "max", "kernel_size": 2},
                {"type": "conv", "in_channels": 16, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
                {"type": "pool", "pool_type": "max", "kernel_size": 2}
            ]
            
            base_fc_config = [(32 * 7 * 7, 128), (128, 10)]
            
            param_grid = {
                "learning_rate": [0.001, 0.01, 0.1],
                "optimizer": ["SGD", "MomentGD"],
                "use_global_avg_pool": [False, True]
            }
            
            configs = study.generate_grid_search_configs(param_grid)
            
            for i, config in enumerate(configs):
                if config["use_global_avg_pool"]:
                    config["fc_configs"] = [(32, 10)]  # Adjusted for global avg pool
                else:
                    config["fc_configs"] = base_fc_config
                    
                config["conv_configs"] = base_conv_config
                config["activation"] = "ReLU"
                config["batch_size"] = 32
                config["num_epochs"] = 5
                config["log_iters"] = 100
                config["scheduler"] = "MultiStepLR"
                config["scheduler_params"] = {"milestones": [30, 60, 90], "gamma": 0.1}
                
                study.run_experiment(config)
            
            study.save_results()
    
    elif args.mode == "single":
        # Run a single experiment with custom parameters
        if args.model == "MLP":
            config = {
                "name": "custom_mlp",
                "hidden_layers": [256],
                "activation": "ReLU",
                "optimizer": "MomentGD",
                "learning_rate": 0.1,
                "momentum": 0.9,
                "scheduler": "ExponentialLR", 
                "scheduler_params": {"gamma": 1.0},
                "batch_size": 32,
                "num_epochs": 100,
                "log_iters": 50
            }
        else:  # CNN
            config = {
                "name": "custom_cnn",
                "conv_configs": [
                        {"type": "conv", "in_channels": 1, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
                        {"type": "pool", "pool_type": "max", "kernel_size": 2}
                    ],
                    "fc_configs": [(16 * 14 * 14, 64),(64,10)],
                    "activation": "ReLU",
                    "use_global_avg_pool": False,
                    "optimizer": "SGD",
                    "learning_rate": 0.1,
                    "scheduler": "ExponentialLR",
                    "scheduler_params": {"gamma": 1.0},
                    "batch_size": 32,
                    "num_epochs": 50,
                    "log_iters": 100
            }
        
        study.run_experiment(config)
        study.save_results()


if __name__ == "__main__":
    main()