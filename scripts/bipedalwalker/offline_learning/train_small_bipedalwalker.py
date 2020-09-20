import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

from nn.mlp import DeepMLPTorch


def load_dataset(features_path: str, labels_path: str):
    X = pd.read_csv(features_path, header=None)
    y = pd.read_csv(labels_path, header=None)
    X = torch.tensor(X.values, dtype=torch.float)
    y = torch.tensor(y.values, dtype=torch.float)
    return X, y


def create_model(input_size, output_size, *hidden_size):
    model = DeepMLPTorch(input_size, output_size, hidden_size)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, opt


def train(train_data, model, opt, loss_func, epochs):
    for epoch in range(epochs):
        for xb, yb in train_data:
            pred = model(xb)
            loss = loss_func(pred, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
        print('Epoch {}: loss = {}'.format(epoch, loss_func(student_model(xb), yb)))


if __name__ == '__main__':
    X, y = load_dataset(
        "../../../models/bipedalwalker/generated_data/features_data_model-layers=24-[20, 12, 12]-4-09-14-2020_11-54_NN=DeepBipedalWalkerIndividual_POPSIZE=40_GEN=5000_PMUTATION_0_NRUNS=10000.csv",
        "../../../models/bipedalwalker/generated_data/labels_data_model-layers=24-[20, 12, 12]-4-09-14-2020_11-54_NN=DeepBipedalWalkerIndividual_POPSIZE=40_GEN=5000_PMUTATION_0_NRUNS=10000.csv")

    BATCH_SIZE = 32
    EPOCHS = 10
    train_ds = TensorDataset(X, y)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    INPUT_SHAPE = 24
    OUTPUT_SHAPE = 4
    HIDDEN_SIZES = [2]
    student_model, opt = create_model(INPUT_SHAPE, OUTPUT_SHAPE, HIDDEN_SIZES)
    loss_func = nn.modules.loss.MSELoss(size_average=False)

    train(train_dl, student_model, opt, loss_func, EPOCHS)
    torch.save(student_model,
               "../../../models/bipedalwalker/large_model/model-layers={}-[{}]-{}-NN=StudentModel.pt".format(
                   INPUT_SHAPE, ','.join(map(str, HIDDEN_SIZES)), OUTPUT_SHAPE))
