class EarlyStopping:
    """Sample usage
    early_stopping = EarlyStopping(tolerance=10, min_delta=0)

    early_stopping(train_loss, eval_loss)
    if early_stopping.early_stop:
        print("Stopped at epoch:", epoch + 1)
        break
    """
    def __init__(self, tolerance=5, min_delta=0):
        """_summary_

        Args:
            tolerance (int, optional): _description_. Defaults to 5.
            min_delta (int, optional): _description_. Defaults to 0.
        """        
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        """_summary_

        Args:
            train_loss (_type_): _description_
            validation_loss (_type_): _description_
        """        
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True