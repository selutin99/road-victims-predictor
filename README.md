# codd-victims-predictor
A neural network with a web application that 
allows you to predict the victims on the roads by parameters.

### The project consists of 2 modules:


#### neural-network module
Include TensorFlow v1.14.0 and Keras v1.0.8
neural network training

#### web-shell module
Enables an interface using the Flask web framework  


### Setup
1. Clone the repository
2. Install Python 3.7.2
3. Install virtualenv use command like: pip install virtualenv
4. In project folder create venv like: virtualenv venv
5. Activate virtualenv like: CALL venv/Scripts/activate
6. Install requirements (in project folder): pip install -r requirements.txt
7. Run the neural network like (change directory to neural_network before): python neural_network.py
8. Copy model.h5 file to webapp
9. Run the webapp like (change directory to webapp before): python app.py