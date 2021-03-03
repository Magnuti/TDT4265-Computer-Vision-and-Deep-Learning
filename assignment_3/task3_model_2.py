import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn, flatten
import torchvision.transforms as transforms
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy


# (conv-relu-pool-batchnorm)
def block(in_channels, out_channels):
    return [
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        ),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.BatchNorm2d(out_channels)
    ]


class Model(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        self.num_classes = num_classes
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            *block(3, 32),
            *block(32, 64),
            *block(64, 128),
        )

        # The output of feature_extractor will be [batch_size, ?, ?, ?]
        # 128 feature maps with 4x4 images, given 32x32 input images
        self.num_output_features = 4*4*128
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        x = self.feature_extractor(x)  # (batch_size, 128, 4, 4)
        x = flatten(x, start_dim=1)  # (batch_size, 128*4*4)
        out = self.classifier(x)  # (batch_size, self.num_classes)
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


if __name__ == "__main__":
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result!
    utils.set_seed(0)
    epochs = 10
    batch_size = 64
    learning_rate = 0.001
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size, data_augmentation=False)
    model = Model(image_channels=3, num_classes=10)
    trainer_with_adam = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders,
        optimizer="Adam"
    )
    trainer_with_adam.train()

    utils.set_seed(0)
    dataloaders = load_cifar10(batch_size, data_augmentation=False)
    model = Model(image_channels=3, num_classes=10)

    trainer_without_adam = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders,
        # optimizer="Adam"
    )
    trainer_without_adam.train()

    plot_path = pathlib.Path("plots")
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer_with_adam.train_history["loss"],
                    label="Training loss with Adam", npoints_to_average=10)
    utils.plot_loss(trainer_without_adam.train_history["loss"],
                    label="Training loss without Adam", npoints_to_average=10)
    utils.plot_loss(
        trainer_with_adam.validation_history["loss"], label="Validation loss with Adam")
    utils.plot_loss(
        trainer_without_adam.validation_history["loss"], label="Validation loss without Adam")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(
        trainer_with_adam.train_history["accuracy"], label="Training Accuracy with Adam")
    utils.plot_loss(
        trainer_without_adam.train_history["accuracy"], label="Training Accuracy without Adam")
    utils.plot_loss(
        trainer_with_adam.validation_history["accuracy"], label="Validation Accuracy with Adam")
    utils.plot_loss(
        trainer_without_adam.validation_history["accuracy"], label="Validation Accuracy without Adam")
    plt.legend()
    plt.savefig(plot_path.joinpath("task3_model_2_plot.png"))
    plt.show()
