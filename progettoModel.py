import unittest
import numpy as np

# Assumiamo che la classe MultimodalStreamingEngine sia definita qui sopra

class TestMultimodalEngine(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """
        setUpClass viene eseguito una sola volta prima di tutti i test.
        Carichiamo i modelli qui per non doverli ricaricare a ogni singolo test (risparmiando molto tempo).
        """
        print("\n--- [TEST] Inizializzazione dei modelli per i test ---")
        cls.engine = MultimodalStreamingEngine()

    def test_chiavi_output_corrette(self):
        """Testa che il dizionario di output contenga sempre le chiavi corrette."""
        # Creiamo 2 secondi di puro rumore bianco (float32)
        dummy_audio = np.random.randn(16000 * 2).astype(np.float32)
        risultato = self.engine.process_live_stream(dummy_audio)
        
        # Verifichiamo la presenza delle chiavi
        self.assertIn("testo", risultato, "Manca la chiave 'testo' nell'output")
        self.assertIn("sentiment_linguistico", risultato, "Manca la chiave 'sentiment_linguistico'")
        self.assertIn("emozione_acustica", risultato, "Manca la chiave 'emozione_acustica'")

    def test_logica_audio_corto(self):
        """
        Testa la condizione di sicurezza: se l'audio è rumore o produce poche parole, 
        l'analisi emozionale e di sentiment non deve partire (ritornando None).
        """
        # Creiamo un array di zeri (silenzio assoluto) di 1 secondo
        silenzio = np.zeros(16000).astype(np.float32)
        risultato = self.engine.process_live_stream(silenzio)
        
        # Whisper solitamente trascrive il silenzio come "" (stringa vuota) o rumore ambientale
        # Dato che la lunghezza delle parole sarà < 3, ci aspettiamo None per le emozioni
        testo_pulito = risultato["testo"].strip()
        
        if len(testo_pulito.split()) <= 3:
            self.assertIsNone(risultato["sentiment_linguistico"], "Il sentiment dovrebbe essere None per audio corti/silenzio")
            self.assertIsNone(risultato["emozione_acustica"], "L'emozione dovrebbe essere None per audio corti/silenzio")
            
    def test_tipo_dati_input(self):
        """Verifica che il modello accetti correttamente numpy array float32."""
        audio_valido = np.array([0.0] * 16000, dtype=np.float32)
        try:
            _ = self.engine.process_live_stream(audio_valido)
            successo = True
        except Exception as e:
            successo = False
            self.fail(f"Il processamento è fallito con un input valido: {e}")
            
        self.assertTrue(successo)

if __name__ == "__main__":
    # Eseguiamo i test
    # Nota: verbosity=2 offre un output più dettagliato nel terminale
    unittest.main(verbosity=2)