import utils
import matplotlib.pyplot as plt
from task2a import pre_process_training_images, one_hot_encode, SoftmaxModel, pre_process_non_training_images
from task2 import SoftmaxTrainer


def plot_and_save(train_history, val_history, filename):
    plt.figure(figsize=(20, 12))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.ylim([0., .5])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.ylim([0.90, 1])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig("{}.png".format(filename))
    plt.show()


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [32, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = True
    use_improved_weight_init = True
    # We reduce the learning rate when using momentum
    use_momentum = True
    learning_rate = 0.02

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train, mean, std = pre_process_training_images(X_train)
    X_val = pre_process_non_training_images(X_val, mean, std)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    plot_and_save(train_history, val_history, "task4a")

    # Set 128 nodes in the hidden layer
    neurons_per_layer = [128, 10]

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    plot_and_save(train_history, val_history, "task4b")
