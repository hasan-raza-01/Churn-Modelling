import torch
import torch.nn as nn
import torch.nn.functional as F


# Define a custom neural network that supports a variable number of hidden layers.
class ClassifierModule(nn.Module):
    """
    A flexible neural network for a classification task that supports a tunable number
    of hidden layers. The network architecture is defined as:
    
    Input -> [Hidden Layer 1 -> ReLU -> Dropout] -> ... -> [Hidden Layer N -> ReLU -> Dropout] -> Output Layer
    
    Attributes:
      input_dim (int): Number of input features.
      num_hidden_layers (int): Number of hidden layers in the network.
      hidden_units (int): Number of neurons in each hidden layer.
      dropout (float): Dropout probability applied after each hidden layer activation.
    """
    def __init__(self, input_dim=20, num_hidden_layers=1, hidden_units=50, dropout=0.5):
        # Initialize the parent nn.Module class.
        super().__init__()
        
        # Create a list to hold our hidden layers. We'll use the ModuleList container so that the layers
        # are registered as submodules (required for proper parameter tracking during training).
        hidden_layers = []
        
        # Add the first hidden layer: From input_dim to hidden_units.
        hidden_layers.append(nn.Linear(input_dim, hidden_units))
        
        # Add additional hidden layers (if any) where each receives hidden_units as input and outputs hidden_units.
        # We subtract one because the first layer is already added.
        for _ in range(num_hidden_layers - 1):
            hidden_layers.append(nn.Linear(hidden_units, hidden_units))
        
        # Save the list of hidden layers in a ModuleList so that it is properly managed.
        self.hidden_layers = nn.ModuleList(hidden_layers)
        
        # Define a Dropout layer applied after each hidden layer activation.
        self.dropout = nn.Dropout(dropout)
        
        # The final output layer maps the last hidden layer's output to the number of classes.
        # Here, 2 is used for binary classification.
        self.output_layer = nn.Linear(hidden_units, 2)
    
    def forward(self, *args, **kwargs):
        """
        Defines the forward pass of the network.
        
        Arguments:
          x (Tensor): Input tensor of shape (batch_size, input_dim)
          
        Returns:
          Tensor: Logits output from the network.
        """
        # If keyword arguments are provided, assume they are features and extract their values.
        # This will convert the keys (e.g., 'CreditScore', …) into a tensor.
        if kwargs:
            # Assuming all columns are numeric and should be concatenated along the feature axis.
            x = torch.tensor([list(sample) for sample in zip(*kwargs.values())])
        else:
            x = args[0]  # usual case if input is a tensor

        # Pass the input through each hidden layer block.
        for layer in self.hidden_layers:
            x = layer(x)       # Apply the linear transformation.
            x = F.relu(x)      # Pass through ReLU activation to introduce non-linearity.
            x = self.dropout(x)  # Apply dropout to reduce overfitting.
        
        # Pass the result from the last hidden layer block into the output layer.
        x = self.output_layer(x)
        # Note: We do not use softmax here; nn.CrossEntropyLoss expects raw logits.
        return x
    
