import models.nn


MODELS = {
    "sven": models.nn.Sven,  # baseline model
    "sven-2": models.nn.Sven2,  # cleaned up implementation of the baseline model
    "bjorn": models.nn.Bjorn,  # speaker embedding
}
