#Clearing all objects
rm(list = ls())

#Installing packages required for image processing and classification
install.packages("rjson")
install.packages("tensorflow")
install.packages("jpeg")
install.packages("readr")
install.packages("h2o")
#________________________________________________________________________________________________________________

#loading libraries required for image processing and classification
library(tensorflow)
library(magrittr)
library(jpeg)
library(readr)
library(grid)
library(ggplot2)
library(stringr)
library("rjson")
library(dplyr)
library(jsonlite)
library(h2o)
#________________________________________________________________________________________________________________

#Step 1
# to start and connect to an H2O instance
localH2Oinstance <- h2o.init(ip = "localhost", port =54321, startH2O = TRUE ,max_mem_size = "25g" )

#Step 2
##Extracting images for each set and preparing the image dataset for classification##

#loading renthop dataset  from the local machine 
json_file<- "C:/Users/tjneh/BAPM/R/train.json"

#the datset obtained is in json format and the logic below is used to convert it into a dataframe
json_str<- paste(readLines(json_file), collapse = "")
json_dat<- fromJSON(json_str)
train = as.data.frame(t(do.call(rbind, json_dat)))


#extracting the colums containing the photos only by using a subset
images = train["photos"]

#as a proof of concept a sample of the data - 100 rows is taken for image classification
#the image for each house is in the form a list - a list containing images of bathroom,kitchen,living room,clostes,foor plan,etc
#the function extractimage extracts the images in the list and creates rows for a house-image combination
#eg: [row 1 col1] = house1, [row1,col2] = image1, [row 2 col1] = house1, [row2,col2] = image2 and so on for each house
extractImages = function(images)
{
  count = 0
  output = data.frame()
  row_total = nrow(images)
  
  for(image_list in 1:100)
  {
#unlisting the images    
    l = unlist(images[image_list,1])
    l = as.vector(l)
#initialising the count variable to get the house id
    count = count + 1
    
    for(image in 1:length(l))
    {
#getting the url link to the image/input "Image not found" in case the url was blank     
      house = cbind("house",count, ifelse(is.null(l[image]),"Image not found",l[image]))
      output = rbind(output, as.data.frame(house))
    }
    
  }
#writing the output to an images file in csv format
  write.csv(output,"C:/Users/tjneh/BAPM/R/images.csv")

}

data = read.csv("C:/Users/tjneh/BAPM/R/images.csv")


#subsetting the data variable to get only the image urls from the column named v3
data = data["V3"]

#________________________________________________________________________________________________________________

#Step 3
#Image classification using slim library and pretrained vvg16 model

imageClassifier = function(image_link)
{
  #TF-slimmodule is a lightweight library for defining, training and evaluating complex models in TensorFlow. 
  #Components of tf-slimmodule can be freely mixed with native tensorflow, as well as other frameworks, 
  #such as tf.contrib.learn.
  #Importing tensorflow.contrib.slimmodule as slimmodule
  slimmodule = tf$contrib$slim
  
  #resetting the default graph
  tf$reset_default_graph()
  
  #the tensor here has an order of 4 - index 1 holds the image number, 2 - width, 3 - height and 4 - color
  #the value 3 ertains to the 3 color channels - r(red), g(green) and b(blue)
  images = tf$placeholder(tf$float32, shape(NULL, NULL, NULL, 3))
  
  #the images of varying size are scaled to the same size
  imgs_scaled = tf$image$resize_images(images, shape(224,224))
  
  #The VGG16 is a convolutional neural network model and the slim library is used to build the network
  #Defining the layers for VGG16 implementation
  # The last layer is the Tensor holding the logits of the classes
  lastlayer = slimmodule$conv2d(imgs_scaled, 64, shape(3,3), scope='vgg_16/conv1/conv1_1') %>% 
    slimmodule$conv2d(64, shape(3,3), scope='vgg_16/conv1/conv1_2')  %>%
    slimmodule$max_pool2d( shape(2, 2), scope='vgg_16/pool1')  %>%
    
    slimmodule$conv2d(128, shape(3,3), scope='vgg_16/conv2/conv2_1')  %>%
    slimmodule$conv2d(128, shape(3,3), scope='vgg_16/conv2/conv2_2')  %>%
    slimmodule$max_pool2d( shape(2, 2), scope='vgg_16/pool2')  %>%
    
    slimmodule$conv2d(256, shape(3,3), scope='vgg_16/conv3/conv3_1')  %>%
    slimmodule$conv2d(256, shape(3,3), scope='vgg_16/conv3/conv3_2')  %>%
    slimmodule$conv2d(256, shape(3,3), scope='vgg_16/conv3/conv3_3')  %>%
    slimmodule$max_pool2d(shape(2, 2), scope='vgg_16/pool3')  %>%
    
    slimmodule$conv2d(512, shape(3,3), scope='vgg_16/conv4/conv4_1')  %>%
    slimmodule$conv2d(512, shape(3,3), scope='vgg_16/conv4/conv4_2')  %>%
    slimmodule$conv2d(512, shape(3,3), scope='vgg_16/conv4/conv4_3')  %>%
    slimmodule$max_pool2d(shape(2, 2), scope='vgg_16/pool4')  %>%
    
    slimmodule$conv2d(512, shape(3,3), scope='vgg_16/conv5/conv5_1')  %>%
    slimmodule$conv2d(512, shape(3,3), scope='vgg_16/conv5/conv5_2')  %>%
    slimmodule$conv2d(512, shape(3,3), scope='vgg_16/conv5/conv5_3')  %>%
    slimmodule$max_pool2d(shape(2, 2), scope='vgg_16/pool5')  %>%
    
    slimmodule$conv2d(4096, shape(7, 7), padding='VALID', scope='vgg_16/fc6')  %>%
    slimmodule$conv2d(4096, shape(1, 1), scope='vgg_16/fc7') %>%
    
    slimmodule$conv2d(1000, shape(1, 1), scope='vgg_16/fc8')  %>%
    tf$squeeze(shape(1, 2), name='vgg_16/fc8/squeezed')
  
  
  #starting a session and extracting the model weights from the vgg_16.ckpt file
  restore = tf$train$Saver()
  new_session = tf$Session()
  restore$restore(new_session, 'C:/Users/tjneh/BAPM/R/vgg_16.ckpt')
  
  
  #Note - the image is loaded from the url and not downloaded so that the latest/updated images are obtained
  #load the image as url and convert into a jpeg format for processing
  image_feed = as.character(image_link)
  z <- tempfile()
  download.file(image_feed,z,mode="wb")
  new_image <- readJPEG(z)
  
  #retrieving the dim attribute
  image_dim = dim(new_image)
  
  #ensuring the images are in the range of 0-255 (rgb color scale) 
  image_array = array(255*new_image, dim = c(1, image_dim[1], image_dim[2], image_dim[3]))
  
  #the prediction is performed by inputing image_array and loading the last_layer having the correct weights
  lastlayer_values = new_session$run(lastlayer, dict(images = image_array))
  
  #sorting the classes with the highest probabilities and loading the imagenet_classes.txt file which contains the description of the classes
  index = sort.int(lastlayer_values, index.return = TRUE, decreasing = TRUE)$ix[1:5]
  probability = exp(lastlayer_values)/sum(exp(lastlayer_values))
  classes = read_delim("C:/Users/tjneh/BAPM/R/imagenet_classes.txt", "\t", escape_double = FALSE, trim_ws = TRUE,col_names = FALSE)
  
  #initialising the class_name variable to contain the name and probabilities of the classification
  class_name = ""
  for (i in index) {
    class_name = paste0(class_name, classes[i,][[1]], " ", round(probability[i],5), "\n") 
  }
  
  #read the values in the class_name string and split it at a new line into different columns
  x = read.table(text = class_name, sep = "\n", colClasses = "character")
  
  #transposing the colums into rows and converting it into a vector
  transpose_x = t(x)
  vect = as.vector(transpose_x)
  vect
  length(vect)
  cha <- NULL
  num <- NULL
  vals <- vector()
  #getting the top 3 classes and probabilities
    for( i in 1:3)
  {
  #the value is in a string format and the data is split into a column containing class and a column containing probability
    cha[i] = gsub("[[:digit:]]","",vect[i])
    num[i] <- as.numeric(str_extract(vect[i], "[0-9]+.[0-9]+"))
    vals = c(vals,cha[i],num[i])
    t(vals)
  }
  
  dframe = as.data.frame(t(vals))
  
  #creating a dataframe with the image link and its classifications
  op_dframe = c(image_feed,dframe)
  op_dframe = as.data.frame(op_dframe)
  
  #adding column names
  colnames(op_dframe) = c("url","class1","pobability1","class2","probability2","class3","probability3")
  
  #output the dataframe
  return(op_dframe)
  
}

#________________________________________________________________________________________________________________

#Step 4

#Writing the classes and the probabilities of each image to an output file

#converting datatype of data to vector for processing
as.vector(data)

#creating a temporary dataframe variable
new_frame = data.frame(url = character(0),c1 = character(0),p1 = numeric(0),c2 = character(0),p2= numeric(0),c3 = character(0),p3= numeric(0))
loopit <- function(data)
{
  for (i in 1:nrow(data))
  {
    #process for rows which have a url only
    if(data[i,1] != 'Image not found')
    {
      op_dframe = imageClassifier(image_link = data[i,1])
      
      new_frame = rbind.data.frame(new_frame,op_dframe)
    }
  }
  return(new_frame)
}

#extract the classification and write it to a csv file
classified_images = loopit(data)
write.csv(classified_images,"C:/Users/tjneh/BAPM/R/image_classification.csv")

#shutdown h2o - Type Y in console to shutdown h2o
h2o.shutdown(prompt = TRUE)

#________________________________________________________________________________________________________________

#the output file contains the classes of each image along with the probabilities

#**********************************************Future roadmap**********************************************************************#
#The classes of the images can be used to see if there is a correlation between the classes and the interest level(target variable)#
#Based on the correlation if any the classes of images which generated maximum interest level can be uploaded to improve the sales #
#**********************************************************************************************************************************#