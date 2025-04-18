import numpy as np
import os
from tqdm import tqdm
import numpy as np
from scipy.ndimage import rotate, shift


class DataAugmenter:
    """
    Class for on-the-fly data augmentation during training with independent probabilities.
    """
    def __init__(self, augmentation_config, model_type="MLP"):
        """
        Initialize the data augmenter.
        
        Args:
            augmentation_config (dict): Configuration for augmentation
            model_type (str): Type of model ('MLP' or 'CNN')
        """
        self.enabled = augmentation_config.get("enabled", False)
        
        # Get probabilities for each augmentation type
        self.rotation_prob = augmentation_config.get("rotation_prob", 0.0)
        self.shift_prob = augmentation_config.get("shift_prob", 0.0)
        self.noise_prob = augmentation_config.get("noise_prob", 0.0)
        
        # Get parameters for each augmentation type
        self.rotation_range = augmentation_config.get("rotation_range", (-30, 30))
        self.shift_range = augmentation_config.get("shift_range", (-5, 5))
        self.noise_level = augmentation_config.get("noise_level", 0.1)
        
        self.model_type = model_type
        
    def augment_batch(self, images, labels):
        """
        Apply augmentation to a batch of data on-the-fly with independent probabilities.
        
        Args:
            images: Batch of images
            labels: Corresponding labels
            
        Returns:
            tuple: Augmented images and labels
        """
        if not self.enabled:
            return images, labels
            

        
        # Create a copy of the images to modify
        augmented_images = images.copy()
        num_samples = len(images)
        
        # Apply each augmentation independently with its probability
        
        # 1. Rotation
        if self.rotation_prob > 0:
            # Determine which samples to rotate
            rotate_mask = np.random.rand(num_samples) < self.rotation_prob
            
            for i in np.where(rotate_mask)[0]:
                angle = np.random.uniform(self.rotation_range[0], self.rotation_range[1])
                
                if self.model_type == "CNN":
                    img_rotated = np.zeros_like(augmented_images[i])
                    img_rotated[0] = rotate(augmented_images[i][0], angle, reshape=False, mode='nearest')
                else:
                    img_2d = augmented_images[i].reshape(28, 28)
                    img_rotated = rotate(img_2d, angle, reshape=False, mode='nearest').reshape(-1)
                
                augmented_images[i] = img_rotated
        
        # 2. Shift
        if self.shift_prob > 0:
            # Determine which samples to shift
            shift_mask = np.random.rand(num_samples) < self.shift_prob
            
            for i in np.where(shift_mask)[0]:
                dx, dy = np.random.uniform(self.shift_range[0], self.shift_range[1], 2)
                
                if self.model_type == "CNN":
                    img_shifted = np.zeros_like(augmented_images[i])
                    img_shifted[0] = shift(augmented_images[i][0], [0, dy, dx], mode='nearest')
                else:
                    img_2d = augmented_images[i].reshape(28, 28)
                    img_shifted = shift(img_2d, [dy, dx], mode='nearest').reshape(-1)
                
                augmented_images[i] = img_shifted
        
        # 3. Noise
        if self.noise_prob > 0:
            # Determine which samples to add noise to
            noise_mask = np.random.rand(num_samples) < self.noise_prob
            
            for i in np.where(noise_mask)[0]:
                noise = np.random.normal(0, self.noise_level, augmented_images[i].shape)
                img_noisy = np.clip(augmented_images[i] + noise, 0, 1)
                augmented_images[i] = img_noisy
        
        return augmented_images, labels

# Modified RunnerM class with on-the-fly augmentation
class RunnerM:
    """
    Runner class with support for on-the-fly data augmentation using independent probabilities.
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None, augmentation_config=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size
        
        # Initialize data augmenter if config is provided
        if augmentation_config and augmentation_config.get("enabled", False):
            model_type = getattr(model, 'model_type', 
                               "CNN" if hasattr(model, 'conv_configs') else "MLP")
            self.augmenter = DataAugmenter(augmentation_config, model_type=model_type)
            print(f"Using data augmentation with settings:")
            for key, val in augmentation_config.items():
                if key != "enabled":
                    print(f"  {key}: {val}")
        else:
            self.augmenter = None

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []
        self.dev_steps = []
    def train(self, train_set, dev_set, **kwargs):
        """
        Train the model with on-the-fly augmentation.
        
        Args:
            train_set: Training data (X, y)
            dev_set: Validation data (X, y)
            **kwargs: Additional arguments (num_epochs, log_iters, save_dir)
        """
        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        best_score = 0
        self.best_score = 0

        for epoch in range(num_epochs):
            X, y = train_set
            iterations_per_epoch = (X.shape[0] + self.batch_size - 1) // self.batch_size
            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(range(X.shape[0]))

            X = X[idx]
            y = y[idx]

            for iteration in range(int(X.shape[0] / self.batch_size) + 1):
                current_step = epoch * iterations_per_epoch + iteration
                train_X = X[iteration * self.batch_size : (iteration+1) * self.batch_size]
                train_y = y[iteration * self.batch_size : (iteration+1) * self.batch_size]

                # Apply on-the-fly augmentation if augmenter is enabled
                if self.augmenter is not None:
                    train_X, train_y = self.augmenter.augment_batch(train_X, train_y)

                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                self.train_loss.append(trn_loss)
                
                trn_score = self.metric(logits, train_y)
                self.train_scores.append(trn_score)

                # the loss_fn layer will propagate the gradients.
                self.loss_fn.backward()

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                
                

                if (iteration) % log_iters == 0:
                    dev_score, dev_loss = self.evaluate(dev_set)
                    self.dev_scores.append(dev_score)
                    self.dev_loss.append(dev_loss)
                    self.dev_steps.append(current_step)
                    print(f"epoch: {epoch}, iteration: {iteration}")
                    print(f"[Train] loss: {trn_loss}, score: {trn_score}")
                    print(f"[Dev] loss: {dev_loss}, score: {dev_score}")

            if dev_score > best_score:
                save_path = os.path.join(save_dir, 'best_model.pickle')
                self.save_model(save_path)
                print(f"best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
                best_score = dev_score
        self.best_score = best_score

    def evaluate(self, data_set):
        """
        Evaluate the model on a dataset.
        
        Args:
            data_set: Evaluation data (X, y)
            
        Returns:
            tuple: (accuracy, loss)
        """
        X, y = data_set
        logits = self.model(X)
        loss = self.loss_fn(logits, y)
        score = self.metric(logits, y)
        return score, loss
    
    def save_model(self, save_path):
        """
        Save the model to disk.
        
        Args:
            save_path: Path to save the model
        """
        self.model.save_model(save_path)