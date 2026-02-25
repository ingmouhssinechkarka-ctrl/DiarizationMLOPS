import os
import torch
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Annotation, Segment

# On importe le pipeline DiariZen
from diarizen.pipelines.inference import DiariZenPipeline

# ==========================================
# 1. CONFIGURATION (Vos dossiers existants)
# ==========================================
AUDIO_DIR = "GroundTruth_overlap_noise/GroundTruth_overlap_noise/audioMono"  # Dossier contenant les fichiers .wav
RTTM_DIR = "GroundTruth_overlap/rttm"
RESULTS_DIR = "GroundTruth_overlap_noise/diarization_results_diarizen"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================
# 2. CHARGEMENT DU MODÈLE
# ==========================================
print("⏳ Chargement de DiariZen (WavLM Large)...")
# Utilisation du GPU si dispo
device = "cuda" if torch.cuda.is_available() else "cpu"

pipeline = DiariZenPipeline.from_pretrained(
    "BUT-FIT/diarizen-wavlm-large-s80-md-v2",
   
)
print(f"✅ Modèle chargé sur {device}")

# ==========================================
# 3. FONCTION UTILITAIRE (Lecture RTTM)
# ==========================================
def load_rttm(file_path):
    """Lit un RTTM Ground Truth et le convertit en Annotation Pyannote"""
    annotation = Annotation()
    if not os.path.exists(file_path):
        return annotation
        
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                start = float(parts[3])
                duration = float(parts[4])
                label = parts[7]
                annotation[Segment(start, start + duration)] = label
    return annotation

# ==========================================
# 4. BOUCLE D'ÉVALUATION
# ==========================================
metric = DiarizationErrorRate()
print("-" * 80)
print(f"{'Fichier':<30} | {'DER (%)':<10} | {'Speakers (Vrai/Pred)'}")
print("-" * 80)

files = sorted([f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")])
der_scores = []

for wav_file in files:
    # Chemins
    wav_path = os.path.join(AUDIO_DIR, wav_file)
    rttm_name = wav_file.replace(".wav", ".rttm")
    rttm_path = os.path.join(RTTM_DIR, rttm_name)
    
    # 1. Inférence DiariZen
    # Note : DiariZen renvoie déjà un objet compatible Pyannote !
    try:
        # On définit un nom de session (sess_name) propre au fichier
        sess_name = wav_file.replace(".wav", "")
        
        # INFERENCE
        hypothesis = pipeline(wav_path, sess_name=sess_name)
        
        # 2. Chargement Ground Truth
        reference = load_rttm(rttm_path)
        
        # 3. Calcul DER (Uniquement si le fichier RTTM existe et n'est pas vide)
        if len(reference) > 0:
            der = metric(reference, hypothesis)
            der_scores.append(der)
            
            # Comptage simple pour affichage
            nb_ref = len(reference.labels())
            nb_hyp = len(hypothesis.labels())
            
            print(f"{wav_file:<30} | {der*100:05.2f}%      | {nb_ref} / {nb_hyp}")
            
            # Optionnel : Sauvegarder le résultat RTTM généré
            with open(os.path.join(RESULTS_DIR, rttm_name), "w") as f:
                hypothesis.write_rttm(f)
                
    except Exception as e:
        print(f"⚠️ Erreur sur {wav_file}: {e}")

# ==========================================
# 5. RÉSULTAT GLOBAL
# ==========================================
if der_scores:
    avg_der = sum(der_scores) / len(der_scores)
    print("-" * 80)
    print(f"🚀 DER MOYEN DIARIZEN : {avg_der * 100:.2f}%")
    print("-" * 80)