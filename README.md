# example-ml

## Train and save model
python train.py

## Start API server
python app/api.py

## Send request (in another terminal or using curl/postman)
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"sepal_length":5.1, "sepal_width":3.5, "petal_length":1.4, "petal_width":0.2}'

##  Check monitor
python monitor.py
