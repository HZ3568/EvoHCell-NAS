import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def mle_loss(y_true, y_pred):
    mean = y_pred[:, 0].unsqueeze(1)
    var = F.softplus(y_pred[:, 1].unsqueeze(1))  # softplus to ensure positivity
    loss = 0.5 * torch.log(2 * torch.tensor(np.pi) * var) + (y_true - mean) ** 2 / (2 * var)
    return torch.mean(loss)


def mape_loss(y_true, y_pred):
    lower_bound = 4.5
    fraction = (y_pred - lower_bound) / (y_true - lower_bound)
    return torch.mean(torch.abs(fraction - 1))


class DenseNet(nn.Module):
    def __init__(self, input_dim, num_layers, layer_width, loss_type='mae', regularization=0.0):
        super(DenseNet, self).__init__()
        self.loss_type = loss_type
        self.regularization = regularization
        layers = []

        layers.append(nn.Linear(input_dim, layer_width))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(layer_width, layer_width))
            layers.append(nn.ReLU())

        self.feature_extractor = nn.Sequential(*layers)

        if loss_type == 'mle':
            self.mean_head = nn.Linear(layer_width, 1)
            self.var_head = nn.Linear(layer_width, 1)
        else:
            self.output_head = nn.Linear(layer_width, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        if self.loss_type == 'mle':
            mean = self.mean_head(features)
            var = F.softplus(self.var_head(features))
            return torch.cat([mean, var], dim=1)
        else:
            return self.output_head(features)


class MetaNeuralnet:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, xtrain, ytrain,
            num_layers=10,
            layer_width=20,
            loss='mae',
            epochs=200,
            batch_size=32,
            lr=0.01,
            verbose=0,
            regularization=0.0,
            **kwargs):

        xtrain = torch.tensor(xtrain, dtype=torch.float32).to(self.device)
        ytrain = torch.tensor(ytrain, dtype=torch.float32).view(-1, 1).to(self.device)

        self.model = DenseNet(
            input_dim=xtrain.shape[1],
            num_layers=num_layers,
            layer_width=layer_width,
            loss_type=loss,
            regularization=regularization
        ).to(self.device)

        if loss == 'mle':
            criterion = mle_loss
        elif loss == 'mape':
            criterion = mape_loss
        else:
            criterion = nn.L1Loss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.99))

        self.model.train()
        for epoch in range(epochs):
            permutation = torch.randperm(xtrain.size(0))
            for i in range(0, xtrain.size(0), batch_size):
                indices = permutation[i:i + batch_size]
                batch_x, batch_y = xtrain[indices], ytrain[indices]

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss_value = criterion(batch_y, outputs)
                loss_value.backward()
                optimizer.step()

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: loss = {loss_value.item():.4f}")

        # Evaluate training error
        self.model.eval()
        with torch.no_grad():
            train_pred = self.model(xtrain)
            if loss == 'mle':
                train_pred = train_pred[:, 0]
            train_pred = train_pred.squeeze().cpu().numpy()
            ytrain_np = ytrain.cpu().numpy().squeeze()
            train_error = np.mean(np.abs(train_pred - ytrain_np))
        return train_error

    def predict(self, xtest):
        xtest = torch.tensor(xtest, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(xtest)
            if preds.shape[1] == 2:
                preds = preds[:, 0].unsqueeze(1)  # return mean only for MLE
            return preds.cpu().numpy()


if __name__ == '__main__':
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    # 生成简单数据
    X, y = make_regression(n_samples=1000, n_features=10, noise=10)
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

    # 训练模型
    model = MetaNeuralnet()
    train_error = model.fit(xtrain, ytrain, loss='mae', verbose=1)
    print(f"Train Error: {train_error:.4f}")

    # 预测
    preds = model.predict(xtest)
    print(f"Prediction Shape: {preds.shape}")
