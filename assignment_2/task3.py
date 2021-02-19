import utils
import matplotlib.pyplot as plt
from task2a import pre_process_training_images, one_hot_encode, SoftmaxModel, pre_process_non_training_images
from task2 import SoftmaxTrainer


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False

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

    use_improved_weight_init = True

    model_improved_weights = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_improved_weights = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_improved_weights, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_improved_weights, val_history_improved_weights = trainer_improved_weights.train(
        num_epochs)

    use_improved_sigmoid = True

    model_improved_sigmoid = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_improved_sigmoid = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_improved_sigmoid, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_improved_sigmoid, val_history_improved_sigmoid = trainer_improved_sigmoid.train(
        num_epochs)

    # We reduce the learning rate when using momentum
    use_momentum = True
    learning_rate = 0.02

    model_with_momentum = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_with_momentum = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_with_momentum, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_with_momentum, val_history_with_momentum = trainer_with_momentum.train(
        num_epochs)

    plt.figure(figsize=(20, 12))

    # Plot training loss
    ax = plt.subplot(1, 2, 1)
    ax.set_title("Training loss")
    plt.ylim([0, .4])
    utils.plot_loss(train_history["loss"], "Original",
                    npoints_to_average=10, plot_variance=False)
    utils.plot_loss(train_history_improved_weights["loss"],
                    "Improved weight initalization", npoints_to_average=10, plot_variance=False)
    utils.plot_loss(train_history_improved_sigmoid["loss"],
                    "Improved sigmoid and improved weight initalization", npoints_to_average=10, plot_variance=False)
    utils.plot_loss(
        train_history_with_momentum["loss"], "With momentum, improved sigmoid and improved weight initalization", npoints_to_average=10, plot_variance=False)
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.legend()

    # Plot validation loss
    ax = plt.subplot(1, 2, 2)
    ax.set_title("Validation loss")
    plt.ylim([0, .4])
    utils.plot_loss(val_history["loss"], "Original", plot_variance=False)
    utils.plot_loss(val_history_improved_weights["loss"],
                    "Improved weight initalization", plot_variance=False)
    utils.plot_loss(val_history_improved_sigmoid["loss"],
                    "Improved sigmoid and improved weight initalization", plot_variance=False)
    utils.plot_loss(
        val_history_with_momentum["loss"], "With momentum, improved sigmoid and improved weight initalization", plot_variance=False)
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.legend()

    plt.savefig("task3_loss.png")
    plt.show()

    plt.figure(figsize=(20, 12))

    # Plot training accuracy
    ax = plt.subplot(1, 2, 1)
    ax.set_title("Training accuracy")
    plt.ylim([0.85, 1])
    utils.plot_loss(train_history["accuracy"], "Original")
    utils.plot_loss(
        train_history_improved_weights["accuracy"], "Improved weight initalization")
    utils.plot_loss(
        train_history_improved_sigmoid["accuracy"], "Improved sigmoid and improved weight initalization")
    utils.plot_loss(
        train_history_with_momentum["accuracy"], "With momentum, improved sigmoid and improved weight initalization")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()

    # Plot validation accuracy
    ax = plt.subplot(1, 2, 2)
    ax.set_title("Validation accuracy")
    plt.ylim([0.85, 1])
    utils.plot_loss(val_history["accuracy"], "Original")
    utils.plot_loss(
        val_history_improved_weights["accuracy"], "Improved weight initalization")
    utils.plot_loss(
        val_history_improved_sigmoid["accuracy"], "Improved sigmoid and improved weight initalization")
    utils.plot_loss(
        val_history_with_momentum["accuracy"], "With momentum, improved sigmoid and improved weight initalization")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig("task3_accuracy.png")
    plt.show()
