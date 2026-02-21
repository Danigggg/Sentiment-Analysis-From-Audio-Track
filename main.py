import torch
# Tensore scalare con valore 7
scalar = torch.tensor(7)

#Dimensione del tensore
#ha dimensione 0 perchè non ha direzione ne spazio, è solo un numero 
scalar.ndim # = 0

#accedere al valore
scalar.item() #valore solo per tensori scalari, funziona solo su essi

#vettore dove il primo è la x e il secondo è la y
vett = torch.tensor([7,7])

#ritorna tensore
print(vett[0].item())


#Matrice
#Rappresentazione grafica 
#4 4 
#3 3
#1 1
matrix = torch.tensor([[4,4]])

print(matrix.ndim)