class EarlyStopping:
    def __init__(self, patience=5, delta=0.5):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, train_loss, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif abs(train_loss - val_loss) > self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
        return self.early_stop
