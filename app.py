import streamlit as ss
import numpy as np #standard
import plotly.express as px  #plots and graphing lib
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from PIL import Image, ImageOps
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.layers import Dense,Input,Flatten
from tensorflow.keras.applications.inception_v3 import InceptionV3,preprocess_input,decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Sample nutrition facts per 100g (adjust as needed)
nutrition_data = {
    "Bread": {"Calories": 265, "Protein": 9, "Carbs": 49, "Fats": 3.2, "Fiber": 2.7, "Sugar": 5, "Vitamin A": 0, "Vitamin C": 0, "Iron": 3.6, "Calcium": 30},
    "Dairy_product": {"Calories": 150, "Protein": 8, "Carbs": 12, "Fats": 8, "Fiber": 0, "Sugar": 11, "Vitamin A": 68, "Vitamin C": 0, "Iron": 0.1, "Calcium": 250},
    "Dessert": {"Calories": 300, "Protein": 3, "Carbs": 50, "Fats": 10, "Fiber": 1, "Sugar": 35, "Vitamin A": 20, "Vitamin C": 0, "Iron": 1.2, "Calcium": 50},
    "Egg": {"Calories": 155, "Protein": 13, "Carbs": 1, "Fats": 11, "Fiber": 0, "Sugar": 1, "Vitamin A": 98, "Vitamin C": 0, "Iron": 1.8, "Calcium": 50},
    "Fried_food": {"Calories": 312, "Protein": 7, "Carbs": 22, "Fats": 22, "Fiber": 1.5, "Sugar": 2, "Vitamin A": 15, "Vitamin C": 2, "Iron": 1.5, "Calcium": 40},
    "Meat": {"Calories": 250, "Protein": 26, "Carbs": 0, "Fats": 17, "Fiber": 0, "Sugar": 0, "Vitamin A": 0, "Vitamin C": 0, "Iron": 3.0, "Calcium": 10},
    "Noodles/Pasta": {"Calories": 157, "Protein": 6, "Carbs": 30, "Fats": 2, "Fiber": 2, "Sugar": 2, "Vitamin A": 0, "Vitamin C": 0, "Iron": 1.5, "Calcium": 15},
    "Rice": {"Calories": 130, "Protein": 2, "Carbs": 28, "Fats": 0.3, "Fiber": 0.4, "Sugar": 0, "Vitamin A": 0, "Vitamin C": 0, "Iron": 0.4, "Calcium": 10},
    "Seafood": {"Calories": 120, "Protein": 20, "Carbs": 0, "Fats": 3, "Fiber": 0, "Sugar": 0, "Vitamin A": 150, "Vitamin C": 2, "Iron": 1.2, "Calcium": 60},
    "Soup": {"Calories": 80, "Protein": 5, "Carbs": 10, "Fats": 3, "Fiber": 1.5, "Sugar": 4, "Vitamin A": 30, "Vitamin C": 10, "Iron": 0.8, "Calcium": 40},
    "veggies/Fruit": {"Calories": 50, "Protein": 2, "Carbs": 12, "Fats": 0.5, "Fiber": 3, "Sugar": 8, "Vitamin A": 400, "Vitamin C": 45, "Iron": 2.0, "Calcium": 50}
}

def dic_maker(arr):
    """Takes in the raw model predictions and returns sorted class-probability pairs"""
    dict_ = {ind: prob for ind, prob in enumerate(arr[0])}

    # Normalize probabilities (optional but can help)
    total_prob = sum(dict_.values())
    if total_prob > 0:
        dict_ = {k: v / total_prob for k, v in dict_.items()}
    
    return sorted(dict_.items(), key=lambda x: x[1], reverse=True)[:3]

def dic_maker_tuple(tuple_arr):
    """Formats model predictions and normalizes probability values."""
    dict_ = {}
    total_prob = sum(prob for _, prob in tuple_arr)  # Normalize probabilities
    for tuple_ in tuple_arr:
        food_name = target_dict[tuple_[0]]
        dict_[food_name] = tuple_[1] / total_prob if total_prob > 0 else 0
    return dict_

def inception_no_gen(image):
  """ 
  prediction happens in this function
  super important, takes in image_path (/content/test_1/test/111.jpg)
  outputs: {1:prob(1),2:prob(2)}
  """
  #image_1 = tensorflow.keras.preprocessing.image.load_img(image_path)

  input_arr = tensorflow.keras.preprocessing.image.img_to_array(image)
  input_arr = preprocess_input(input_arr)
  input_arr = tensorflow.image.resize(input_arr,size = (256,256))
  input_arr = np.array([input_arr])  # Convert single image to a batch.
  predictions = model_saved.predict(input_arr)
  return dic_maker_tuple(dic_maker(predictions))

def plot_pred_final(test_imgs):
    """
    Takes in {1:prob(1),2:prob(2)}
    and plots a bar chart along with the uploaded image.
    """
    fig = make_subplots(rows=2, cols=2)

    pred_list = inception_no_gen(test_imgs)
    # Add bar chart of predictions
    fig.add_trace(
        go.Bar(
            y=list(pred_list.keys()), 
            x=list(pred_list.values()), 
            orientation='h', 
            marker=dict(color='tomato')  # Add color for clarity
        ), 
        row=1, 
        col=2
    )
    # Improve visualization
    fig.update_xaxes(title_text="Prediction Confidence", row=1, col=2)  # X-axis label
    fig.update_yaxes(title_text="Food Categories", row=1, col=2)  # Y-axis label
    # Improve visualization
    fig.update_xaxes(range=[0, 1])  
    fig.update_layout(
        width=1000, 
        height=500, 
        title_text="Custom Predictions", 
        showlegend=False,
        font=dict(size=15),
    )

    return fig

def estimate_nutrition(predictions):
    """Estimates nutrition content based on predicted food probabilities."""
    estimated_nutrition = {"Calories": 0, "Protein": 0, "Carbs": 0, "Fats": 0, "Fiber": 0, "Sugar": 0, "Vitamin A": 0, "Vitamin C": 0, "Iron": 0, "Calcium": 0}
    
    for food, prob in predictions.items():
        if food in nutrition_data:
            for key in estimated_nutrition:
                estimated_nutrition[key] += nutrition_data[food][key] * prob

    return estimated_nutrition

#------streamlit starts here----------------

model_saved = tensorflow.keras.models.load_model("inception_food_rec_50epochs.h5")
target_dict = {0:"Bread",1:"Dairy_product",2:"Dessert",3:"Egg",4:"Fried_food",
                 5:"Meat",6:"Noodles/Pasta",7:"Rice",8:"Seafood",9:"Soup",10:"veggies/Fruit"}
ss.set_page_config(page_title = "Food Recognition using Inception V3", layout = "wide")
ss.title("Food Recognition using inception-V3")

ss.markdown(
'''
Every one likes food! This deployment recognizes 11 different classes of food using a SOTA Inception V3 Transfer Learning.\n
''')

ss.image("f1.jpg")
ss.markdown(
'''
### Inception V3
- The paper for Inception can be found [here](https://arxiv.org/abs/1512.00567v3)\n

- The paper implementation using pytorch can be found [here](https://github.com/pytorch/vision/blob/6db1569c89094cf23f3bc41f79275c45e9fcb3f3/torchvision/models/inception.py#L64)

- Inception-v3 is a convolutional neural network architecture from the Inception family that makes several improvements including using 
  - Label Smoothing,
  - Factorized 7 x 7 convolutions,\n 
  and the use of an auxiliary classifer to propagate label information lower down the network (along with the use of batch normalization for layers in the sidehead).
- Training on 16,600 images yielded 90% accuracy on train and 76% accuracy on validation. over 50 epochs!
- This model is saved and used later
'''
)
ss.markdown(
'''
### Model Architecture
''')
ss.image("inception_2.png")

ss.markdown('### Dataset Details and Classes')
ss.markdown('Data consists of 1.1GB of 16,600 images of different categories of food.')
ss.markdown('the categories of food that can be classified are ')
ss.markdown(
  '''
    - Bread
    - Dairy Product
    - Dessert
    - Egg
    - Fried Food
    - Meat
    - Noodles-pasta
    - Rice
    - Seafood
    - Soup
    - Vegetable-fruit
  '''
)
ss.markdown('Dataset is obtained from [kaggle](https://www.kaggle.com/trolukovich/food11-image-dataset)')


ss.markdown('### Food Recognition step - Upload Image')
image_path = ss.file_uploader("drop the image file here: ", type = ["jpg"])

if image_path:
    image = Image.open(image_path)
    
    # Resize the image to a smaller size (e.g., 300x300 pixels)
    max_size = (300, 300)
    image.thumbnail(max_size)  # This resizes the image while maintaining aspect ratio
    # Add the double border effect
    border_thickness_black = 1   # Thin black border
    border_thickness_white = 4   # Thicker white border
    border_thickness_black_outer = 2  # Outer black border

    # Apply borders step by step
    image = ImageOps.expand(image, border=border_thickness_black, fill="black")
    image = ImageOps.expand(image, border=border_thickness_white, fill="white")
    image = ImageOps.expand(image, border=border_thickness_black_outer, fill="black")

    ss.image(image, caption="Uploaded Image", use_column_width=False)  # Set use_column_width=False to prevent it from stretching

    preds = plot_pred_final(image)
        # Estimate nutrition content
    nutrition_estimates = estimate_nutrition(inception_no_gen(image))
    nutrition_estimates = {key: round(value) for key, value in nutrition_estimates.items()}
    # Display estimated nutrition facts
    ss.markdown("### Estimated Nutrition Per 100g")
    ss.json(nutrition_estimates)
    ss.plotly_chart(preds, use_container_width=True)