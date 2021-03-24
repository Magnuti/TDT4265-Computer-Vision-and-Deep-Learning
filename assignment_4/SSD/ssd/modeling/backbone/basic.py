from torch import nn


class BasicModel(nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """

    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS

        # Note that these layers must be saved to self variables because the cfg uses them
        self.block1 = nn.Sequential(
            nn.Conv2d(image_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.output_channels[0], 3, stride=2,  padding=1),
        )

        self.block2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.output_channels[0], 128, 3, padding=1),
            nn.BatchNorm2d(128),

            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),

            nn.ReLU(),
            nn.Conv2d(128, self.output_channels[1], 3, stride=2, padding=1),
        )

        self.block3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.output_channels[1], 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, self.output_channels[2], 3, stride=2, padding=1),
        )

        self.block4 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.output_channels[2], 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, self.output_channels[3], 3, stride=2, padding=1),
        )

        self.block5 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.output_channels[3], 128, 3, padding=1),
            nn.BatchNorm2d(128),

            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),

            nn.ReLU(),
            nn.Conv2d(128, self.output_channels[4], 3, stride=2, padding=1),
        )

        self.block6 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.output_channels[4], 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),

            nn.ReLU(),
            nn.Conv2d(128, self.output_channels[5], 3, padding=0),
        )

        self.output_feature_maps = [
            self.block1, self.block2, self.block3, self.block4, self.block5, self.block6]

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        for output_feature_map in self.output_feature_maps:
            output = output_feature_map(x)  # Pass x through the layer
            out_features.append(output)
            x = output

        for i, feature in enumerate(out_features):
            w, h = self.output_feature_shape[i]
            expected_shape = (self.output_channels[i], h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output index: {i}"
        return tuple(out_features)
