# music_genre_classification
music genre classification : GRU vs Transformer


### Data:

<https://github.com/mdeff/fma>

### Steps to install env:

```
python -m pip install -r requirements
```

### Steps to run:

* Uncompress the data zips (fma_metadata.zip, fma_large.zip).
* Run [prepare_data.py](code/prepare_data.py) with the correct paths to genrate mapping files.
* Run [audio_processing.py](code/audio_processing.py) with the correct paths to genrate .npy files.
* Run training with [rnn_genre_classification.py](code/rnn_genre_classification.py) or [trsf_genre_classification.py](code/trsf_genre_classification.py)

* To predict on new mp3s run [predict.py](code/predict.py) with the correct paths.