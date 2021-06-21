# Ukraine_used_cars_price_prediction
The  system predicts approximate price of used cars in Ukraine.

## Algorithm
The system uses LGBMRegressor.
Best params for regressor were computed using bayesian optimization.

## Data
All  data are scraped from <a href='https://auto.ria.com/uk/'>autoria.com</a> using Scraper class.
You can find all data in Data folder


## Set up
* Install Python 3.8 
* Create and activate virtual environment
* `pip install -U -r requirements.txt`
* Configure data paths and other settings in `config.yml`
* Run training via `python User_pipeline.py`

## Training
The whole training pipeline can be found in [User_pipeline](train_pipeline.py).

