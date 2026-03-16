import os
import shutil

CARTELLA_SORGENTE = "/Users/denisye/Downloads/dataSetEmozioni/" 
CARTELLA_DESTINAZIONE = "/Users/denisye/Downloads/dataSetEmozioniPulito/"

def organizza_dataset_ravdess(percorso_sorgente, percorso_destinazione):
    dizionario_emozioni = {"01": "neutrale", "02": "calmo", "03": "felice", "04": "triste", "05": "arrabbiato",
                           "06": "impaurito", "07": "disgustato"," 08": "sorpreso"}

    
    for emozione in dizionario_emozioni.values():
        os.makedirs(os.path.join(percorso_destinazione, emozione), exist_ok=True)
    
        
    file_spostati = 0
    
    for radice, cartelle, file in os.walk(percorso_sorgente):
        for nome_file in file:
            if nome_file.endswith(".wav"):
                parti = nome_file.split('-')
                if len(parti) == 7: 
                    codice_emozione = parti[2]
                    nome_emozione = dizionario_emozioni.get(codice_emozione)
                    
                    if nome_emozione:
                        percorso_origine = os.path.join(radice, nome_file)
                        percorso_destinazione_file = os.path.join(percorso_destinazione, nome_emozione, nome_file)
                        
                        shutil.copy2(percorso_origine, percorso_destinazione_file)
                        file_spostati += 1


if __name__ == "__main__":
    if os.path.exists(CARTELLA_SORGENTE):
        organizza_dataset_ravdess(CARTELLA_SORGENTE, CARTELLA_DESTINAZIONE)
    else:
        print(f"ERRORE: La cartella sorgente '{CARTELLA_SORGENTE}' non esiste. Controlla il percorso.")