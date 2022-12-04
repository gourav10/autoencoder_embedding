import torch

DATA_DIR = "data"

NUM_WORKERS = 0

# how many samples per batch to load
BATCH_SIZE = 20
NUM_EPOCHS = 30
EPOCHS = 10

Loss_fns = {
    "MSE": torch.nn.MSELoss(),
    "CrossEntropy": torch.nn.CrossEntropyLoss(),
    "TripletLoss": torch.nn.TripletMarginLoss(),
    "HingeLoss": torch.nn.HingeEmbeddingLoss()
}
