import torch

# Tensore scalare con valore 7
scalar = torch.tensor(7)

# Dimensione del tensore
# Ha dimensione 0 perchè non ha direzione ne spazio, è solo un numero 
scalar.ndim # = 0

# Accedere al valore
scalar.item() #valore solo per tensori scalari, funziona solo su essi

# Vettore dove il primo è la x e il secondo è la y
vett = torch.tensor([7,7])

# Ritorna tensore
vett[0].item() 

#Cubo
CUBE = torch.tensor([[[4,4],
                        [5,5]],
                        [[5,6],[6,7]]])


# Tensori con valori random
# importanti perchè molte reti neurali imparano
# partendo da valori randomici, 
# aggiustandoli per rappresentare meglio i dati 
# parti con valori random -> guarda i dati -> aggiorna i numeri -> guarda i dati ...
random_tensor = torch.rand(3,4) #crea una matrice 3x4 con valori tra [0,1]

# Creiamo un tensore con una forma simile a un immagine tensore
random_image = torch.rand(224,224,3) #altezza, lunghezza, colori (R,G,B)s

# Tensore con tutti 0
zero_tensor = torch.zeros(4)


# I tensori devono avere preferibilmente lo stesso tipo, 
# 2 tensori su due disposivi diversi sono incompatibili
f = torch.tensor([3,4,5,6,7],dtype=torch.float32, device="cuda")

# Operazioni sui tensori
# addizioni
# sottrazioni
# moltiplicazioni 
# divisioni
# moltiplicazioni tra matrici
torch.manual_seed(7)

ten = torch.rand(1,1,1,10)
ten1 = torch.squeeze(ten)


from torch import nn #neural networks
import matplotlib.pyplot as plt

x = torch.arange(0,1,0.02).unsqueeze(dim=1)
y = 0.7 * x + 0.3

train_split = int(0.8*len(x))
x_train, y_train = x[:train_split], y[:train_split]
x_test, y_test = x[train_split:], y[train_split:]

# Visualizzare i dati

def plot_pred(train_data,train_labels,test_data,test_labels, predictions):
    plt.figure(figsize=(10,7))
    # Raffigura i dati
    plt.scatter(train_data,train_labels,c="b", s=4,label="Training Data")
    plt.scatter(test_data,test_labels,c="g", s=4,label="Testing Data")

    if predictions is not None:
        plt.scatter(test_data,predictions)
    plt.legend()
    plt.show()



# Costruiamo il primo modello
# Cosa fa? Parte con valori randomici, guarda i dati di allenamento, e aggiusta i valori randomici per avvicinarsi
# Come lo fa? Con algoritmi come Gradient descent e back propagation
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # weight è un parametro assegnato inizialmente a un valore casuale (torch.randn(1)) ed è un parametro della neural network
        self.weight = nn.Parameter(torch.randn(1,requires_grad=True,dtype=torch.float)) 
        
        # stessa cosa per bias
        self.bias = nn.Parameter(torch.randn(1,requires_grad=True,dtype=torch.float))

    # Se erediti da nn.Module dove fare overriding di forward()
    def forward(self,x:torch.Tensor) -> torch.Tensor: # x sono i dati in input
        return self.weight * x + self.bias

model_0 = LinearRegressionModel()

print(model_0.state_dict())