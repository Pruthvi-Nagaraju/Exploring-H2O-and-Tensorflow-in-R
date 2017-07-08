#Clearing all objects
rm(list = ls())

#Installing packages required
install.packages("h2o")
#________________________________________________________________________________________________________________

#loading libraries required

library(h2o)
library(tm)
library(jsonlite)
library(dplyr)
library(data.table)
#________________________________________________________________________________________________________________

# to start and connect to an H2O instance
localH2Oinstance <- h2o.init(ip = "localhost", port =54321, startH2O = TRUE ,max_mem_size = "25g" )


json_file<- "C:/R/Project/train.json"

#loading the json file
json_data<- fromJSON(json_file)

#corce json_data into a dataframe
train <- as.data.frame(t(do.call(rbind, json_data)))

View(train)
class(train)

class(train$features)
class(train$bathrooms)
# all the columns in dataset are "list"

#Since we are not focusing on the accuracy of the model and just want to show model building in H2O
#We are not using the features and photos,instead for simplicity purpose we just use 
#the number of features and photos as predictors

numberOfFeatures <- as.numeric(lapply(train$features, length))
numberOfPhotos <- as.numeric(lapply(train$features, length))

#unlisting each column and converting it into data table

table1 <- data.table( bathrooms=unlist(train$bathrooms)
                 ,bedrooms=unlist(train$bedrooms)
                 ,building_id=as.factor(unlist(train$building_id))
                 ,created=as.POSIXct(unlist(train$created))
                 ,num_features=as.numeric(lengths(train$features))
                 ,num_photos=as.numeric(lengths(train$photos))
                 ,latitude=unlist(train$latitude)
                 ,longitude=unlist(train$longitude)
                 ,listing_id=unlist(train$listing_id)
                 ,manager_id=as.factor(unlist(train$manager_id))
                 ,price=unlist(train$price)
                 ,interest_level=as.factor(unlist(train$interest_level)))

# column "created" is of type POSIXct, which cannot be directly used in the model 
# hence extracting year, month, day and hour separately

table2 <- table1[,":="(yday=yday(created)
            ,month=month(created)
            ,mday=mday(created)
            ,wday=wday(created)
            ,hour=hour(created))]

# as.h2o is used to import R object to the H2O cloud
train <- as.h2o(table2[,-"created"], destination_frame = "train.hex")
#________________________________________________________________________________________________________________

#Step 1
#splitting the data

#splitting data into train, validation and test (70,20,10)
splits <- h2o.splitFrame(data = train, ratios = c(0.7,0.2),   
                         destination_frames = c("train.hex", "valid.hex", "test.hex"))

#assigning each of it to train, valid and test
train=splits[[1]]
valid=splits[[2]]
test=splits[[3]]
#________________________________________________________________________________________________________________


#Step 2 
#Establish baseline performance

varnames <- setdiff(colnames(train), "interest_level")

gbm1 <- h2o.gbm(x= varnames, y="interest_level", training_frame = train )

summary(gbm1)

#Confusion Matrix: vertical: actual; across: predicted
#       high   low medium  Error             Rate
#high    523  1036    753 0.7738 =  1,789 / 2,312
#low      39 19906    735 0.0374 =   774 / 20,680
#medium  114  4722   1893 0.7187 =  4,836 / 6,729
#Totals  676 25664   3381 0.2489 = 7h2o.s,399 / 29,721


gbm2 <- h2o.gbm(x= varnames, y="interest_level", training_frame = train , validation_frame = valid,
                ntrees = 1000, learn_rate = 0.01, stopping_metric = "misclassification")

summary(gbm2)
plot(gbm2)

#ntrees=1000, learn_rate = 0.01, 
#Confusion Matrix: vertical: actual; across: predicted
#        high   low medium  Error             Rate
#high    806   745    720 0.6451 =  1,465 / 2,271
#low      40 19828    764 0.0390 =   804 / 20,632
#medium  101  3706   2907 0.5670 =  3,807 / 6,714
#Totals  947 24279   4391 0.2052 = 6,076 / 29,617


TestModel<- h2o.predict(gbm2, newdata = test )
summary(TestModel)

#________________________________________________________________________________________________________________

#Step 3
#Implementation of Random Forest

rf1  <- h2o.randomForest(training_frame = train, validation_frame = valid, x=varnames,y="interest_level",                          
                         model_id = "rf1", ntrees = 1000)

summary(rf1)

plot(rf1)

#Confusion Matrix: vertical: actual; across: predicted
#       high  low medium  Error          Rate
#high    445   19      0 0.0409 =    19 / 464
#low       0 4114      0 0.0000 =   0 / 4,114
#medium    1  106   1252 0.0787 = 107 / 1,359
#Totals  446 4239   1252 0.0212 = 126 / 5,937
#________________________________________________________________________________________________________________

#Step 4 
#Implementation of Deep Learning

deepLearningModel <- h2o.deeplearning(x=varnames, y= "interest_level", training_frame = train,
                                      hidden=c(32,32,32),validation_frame = valid, epochs = 100)

summary(deepLearningModel)
plot( deepLearningModel)

#  Confusion Matrix: vertical: actual; across: predicted
#       high  low medium  Error          Rate
#high    449    7     32 0.0799 =    39 / 488
#low       3 4109     48 0.0123 =  51 / 4,160
#medium   10   10   1287 0.0153 =  20 / 1,307
#Totals  462 4126   1367 0.0185 = 110 / 5,955


#Hyper Parameter search for GBM
#using a small sample of data for demonstration of hyper parameter search
#to find the best value for max_depth:

testData <- train[1:1000,]
hyper_params = list( max_depth = seq(1,10,2))

grid <- h2o.grid( hyper_params = hyper_params, search_criteria = list(strategy = "Cartesian"),
                  algorithm="gbm", grid_id="depth_grid", x = varnames, y = "interest_level", 
                  training_frame = testData, ntrees = 100,                                                            
                  learn_rate = 0.05)                            
grid
#Hyper-Parameter Search Summary: ordered by increasing logloss
#max_depth          model_ids             logloss
#1         9 depth_grid_model_4 0.07609929602578024
#2         7 depth_grid_model_3 0.14258568985051848
#3         5 depth_grid_model_2   0.251431789685268
#4         3 depth_grid_model_1  0.4658812251362307
#5         1 depth_grid_model_0  0.6695233529783211                                                                 

# we can sort the grid models in decreasing order of mean_per_class_error
sortedGrid <- h2o.getGrid(grid@grid_id, sort_by="mean_per_class_error", decreasing = TRUE)    

#shutdown h2o - Type Y in console to shutdown h2o
h2o.shutdown(prompt = TRUE)


#https://drive.google.com/open?id=0B0pAq8YHeKVdNTNOTm1WWnZQTVk
#Mentioned above is the link to a 5GB dataset used to illustrate the code 

#how H2O efficiently manages the importing of it.
#file=read.csv("C:/R/Project/New folder/5gb/c1.csv")
#Normal read.csv throws an error, as the R environment fails in dealing with huge datasets.

#bigdata.hex = h2o.importFile( path = "C:/R/Project/New folder/5gb/c1.csv", destination_frame = "bigdata.hex")
#h2o.importFile function in H2O allows for smooth loading of this huge data in no time.






