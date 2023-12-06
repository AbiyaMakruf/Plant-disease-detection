from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def predict(imageFile,model):
    pred = ['Apple Scab','Apple Black Rot','Apple Cedar/Apple Rust','Apple Healthy','Potato Early blight','Potato Late blight','Potato healthy']

    #image = Image.open(imageFile)
    #np.array(Image.fromarray(np.array(image)).resize((150, 150)))
    
    img = image.load_img(imageFile, target_size=(150, 150))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])


    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test = datagen.flow(images)

    # Making the predictions
    classes = model.predict(test)
    predictions_arr = np.argmax(classes)

    # Convert confidence scores to percentages
    confidence_percentages = [float(score) * 100 for score in classes[0]]

    return {
        "prediction": pred[predictions_arr],
        "confidence": f"{confidence_percentages[predictions_arr]:.2f}%",
    }


#Kode untuk menampilkan confidence score dari semua prediksi 
#results = [{"class": pred[i], "confidence": f"{confidence_percentages[i]:.2f}%"} for i in range(len(pred))]