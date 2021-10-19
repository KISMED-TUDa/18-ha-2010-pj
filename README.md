# 18-ha-2010-pj
Demo-Code zum Projektseminar "Wettbewerb künstliche Intelligenz in der Medizin" WiSe 2021/2022. Das Beispiel definiert gleichzeitig das Interface zu unserem Evaluierungs-System.

## Erste Schritte

1. Klone/Forke dieses Repository
2. Richte ein eigenes Repository auf github/gitlab ein. Darüber könnt ihr später die Abgaben eurer Modelle machen.
3. Python Environment anlegen (z.B. mit Anaconda), dann kann "requirements.txt" mit `pip install -r requirements.txt` ausgeführt werden und installiert die notwendigen Pakete 

## Wichtig!

Die Dateien 
- predict_pretrained.py
- wettbewerb.py
- score.py

werden von uns beim testen auf den ursprünglichen Stand zurückgesetzt. Es ist deshalb nicht empfehlenswert diese zu verändern. In predict.py ist für die Funktion `predict_labels` das Interface festgelegt, das wir für die Evaluierung verwenden.

`predict_labels(ecg_leads : List[np.ndarray], fs : float, ecg_names : List[str], model_name : str='model.npy',is_binary_classifier : bool=False) -> List[Tuple[str,str]]`

Insbesondere den `model_name` könnt ihr verwenden um bei der Abgabe verschiedene Modelle zu kennzeichnen, welche zum Beispiel durch eure Ordnerstruktur dargestellt werden. Der Parameter `is_binary_classifier` ermöglicht es zu entscheiden, ob mit dem Modell nur die zwei Hauptlabels "Atrial Fibrillation ['A']" und "Normal ['N']" klassfiziert werden (binärer Klassifikator), oder alle vier Label.

Bitte gebt alle verwendeten packages in "requirements.txt" bei der Abgabe zur Evaluation an und testet dies vorher in einer frischen Umgebung mit `pip install -r requirements.txt`. Als Basis habt ihr immer die vorgegebene "requirements.txt"-Datei. Wir selbst verwenden Python 3.8. Wenn es ein Paket gibt, welches nur unter einer anderen Version funktioniert ist das auch in Ordung. In dem Fall bitte Python-Version mit angeben.
