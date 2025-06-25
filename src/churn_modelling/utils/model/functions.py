from churn_modelling.exception import CustomException
from churn_modelling.utils.model import ClassifierModule
from skorch import NeuralNetClassifier
import torch.nn as nn
import torch, sys 



def get_NeuralNetClassifier(
        module=ClassifierModule, 
        criterion=nn.CrossEntropyLoss, 
        optimizer=torch.optim.Adam, 
        optimizer__weight_decay=0.01, 
        max_epochs=10, 
        lr=0.01, 
        batch_size=64, 
        module__input_dim=13, 
        module__num_hidden_layers=1, 
        module__hidden_units=50, 
        module__dropout=0.5, 
        iterator_train__shuffle=True, 
        device='cuda' if torch.cuda.is_available() else 'cpu', 
        verbose=1 
    ) -> NeuralNetClassifier:
    try:
        return NeuralNetClassifier(
            module=module,
            criterion=criterion,
            optimizer=optimizer,
            optimizer__weight_decay=optimizer__weight_decay,
            max_epochs=max_epochs,
            lr=lr,
            batch_size=batch_size,
            module__input_dim=module__input_dim,
            module__num_hidden_layers=module__num_hidden_layers,
            module__hidden_units=module__hidden_units, 
            module__dropout=module__dropout,
            iterator_train__shuffle=iterator_train__shuffle,
            device=device,
            verbose=verbose
        )
    except Exception as e:
        raise CustomException(e, sys)
    
