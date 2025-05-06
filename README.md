# Een experiment runnen
De huidige staat van de repository (en code op de cluster) maakt vooral gebruik van de `experiment_config.py` file.
- Zet een unieke `self.EXPERIMENT_NAME`, anders worden oude resultaten mogelijk overschreven
- Configureer de juiste `self.loss_function`
- Pas de beschikbare hyperparameters aan in de file
- submit naar de cluster met `sbatch experiments/exp_baseline_[2/3]D`

## Model architectuur aanpassen
Kopiëer of voeg een ResNet class toe in de `model_2D.py` file en importeer deze vervolgens in `experiment_config.py`
- Zet het model actief als `self.model`

Het 3D model is een stuk complexer, volg dezelfde stappen of bij grote aanpassingen kopiëer de hele `model_3D.py` file en importeer de aangepaste onder de nieuwe naam.

## Resultaat loggen
Voeg een nieuwe row aan de LUNA Log Sheet toe (https://docs.google.com/spreadsheets/d/1qgvJVBFYyHRHY5AuRHHy6PvaNs8IGyccNpVK1vAkuzQ/edit?usp=sharing)
met daarin de relevante namen en waardes. 

De op de cluster gerunde experimenten hebben automatisch ook de log en result files staan op de juiste plek,
plaats bij lokaal gerunde experimenten de terminal output in een text file om het later terug te kunnen lezen.

# Een aanpassing aan de code maken
De huidige repository staat werkt, om dat langer zou te houden s.v.p aanpassingen eerst in je eigen branch maken en als ze compleet en een verbetering zouden zijn:
- Maak een `Pull request` voor jouw branch -> main
- Resolve merge conflicts
- Iemand anders checkt dat de code zal blijven werken en keurt de merge goed/af
  
