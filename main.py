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
MATRIX = torch.tensor([[[4,4],
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

print(f)