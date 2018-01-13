# Code is picked from the link
# https://www.kaggle.com/tannercarbonati/detailed-data-analysis-ensemble-modeling

require(ggplot2) # for data visualization
require(stringr) #extracting string patterns
require(Matrix) # matrix transformations
require(glmnet) # ridge, lasso & elastinet
require(xgboost) # gbm
require(randomForest)
require(Metrics) # rmse
require(dplyr) # load this in last so plyr doens't overlap it
require(caret) # one hot encoding
require(scales) # plotting $$
require(e1071) # skewness
require(corrplot) # correlation plot

setwd("D:\\Himanshu\\Acads\\03. Kaggle\\House Price prediction")


train <- read.csv('train.csv', stringsAsFactors = F)
test <- read.csv('test.csv', stringsAsFactors = F)

## remove ID and SalePrice using with in command

dataset <- rbind(within(train, rm("Id","SalePrice")), within(test, rm("Id")))

dim(dataset)
# [1] 2919   79
# 
# > table(colSums(is.na(dataset)))
# 
# 0    1    2    4   23   24   79   80   81   82  157  159  486 1420 2348 2721 2814 2909 
# 45   11    4    1    1    1    1    1    1    2    1    4    1    1    1    1    1    1 


# Select columns that have sum of NA values > 0

nacols <- which(colSums(is.na(dataset))>0)

# colSums gives a numeric vector that has names as it's attributes. we can say it as a lable to the numeric output vector.
# which function would filter the numeric vector and output indexes of the columns


# Sort NA column values
sort(colSums(is.na(dataset[nacols])), decreasing = T)

# Using paste 

paste('There are', length(nacols), 'cols with missing values')
# [1] "There are 34 cols with missing values"

# Plot some categorical columns
qplot(na.omit(dataset[,"PoolQC"])) + geom_bar(fill = 'cornflowerblue') + geom_text(aes(label = ..count..), stat='count', vjust=-0.5)
+ xlab(col)

# Writing a full function

plot.categoric <- function(cols, df){
  for (col in cols) {
    order.cols <- names(sort(table(dataset[,col]), decreasing = TRUE))
    
    num.plot <- qplot(df[,col]) +
      geom_bar(fill = 'cornflowerblue') +
      geom_text(aes(label = ..count..), stat='count', vjust=-0.5) +
      theme_minimal() +
      scale_y_continuous(limits = c(0,max(table(df[,col]))*1.1)) +
      scale_x_discrete(limits = order.cols) +
      xlab(col) +
      theme(axis.text.x = element_text(angle = 30, size=12))
    
    print(num.plot)
  }
}

plot.categoric("PoolQC", dataset)

dataset[(dataset$PoolArea>0 & is.na(dataset$PoolQC)),c('PoolQC','PoolArea')]

# calculate mean, count by each group of poolQC

dataset[, c('PoolQC','PoolArea')] %>% group_by(PoolQC) %>% summarise(mean = mean(PoolArea), count=n())

dataset[2421, 'PoolQC'] = 'Ex'
dataset[2504, 'PoolQC'] = 'Gd'
dataset[2600, 'PoolQC'] = 'Gd'

dataset$PoolQC[is.na(dataset$PoolQC)] = 'None'

length(which(dataset$GarageYrBlt == dataset$YearBuilt))
# 2216

sum(is.na(dataset$GarageYrBlt))
# 159

# Finding indexes of NA value of this column
idx <- which(is.na(dataset$GarageYrBlt))
dataset[idx, 'GarageYrBlt']  <-  dataset[idx, 'YearBuilt']

table(dataset$GarageCond)
garage.cols <- c('GarageArea', 'GarageCars', 'GarageQual', 'GarageFinish', 'GarageCond', 'GarageType')
# Index 2127 has Garage Area = 360 but Garage condition is NA. so there is some error in the data. 
# use below method to attach frequent values 

dataset[is.na(dataset$GarageCond),garage.cols]
idx <- which(((dataset$GarageArea < 370) & (dataset$GarageArea > 350)) & (dataset$GarageCars == 1))

names(sapply(dataset[idx, garage.cols], function(x) sort(table(x), decreasing=TRUE)[1]))

dataset[2127, 'GarageQual'] = 'TA'
dataset[2127, 'GarageFinish'] = 'unf'
dataset[2127, 'GarageType'] = 'Attchd'
dataset[2127, 'GarageCond'] = 'TA'

dataset[2127, garage.cols]

for (col in garage.cols) {
  if(sapply(dataset[col], is.numeric)==TRUE){
    dataset[sapply(dataset[col], is.na), col]=0
  }  
  else {
    dataset[sapply(dataset[col], is.na), col]="none"
  }
}


plot.categoric('KitchenQual',  dataset)
# Assign NA to value TA
dataset$KitchenQual[is.na(dataset$KitchenQual)]='TA'

plot.categoric('Electrical', dataset)
sum(is.na(dataset$Electrical))
dataset$Electrical[is.na(dataset$Electrical)]='SBrkr'

bsmt.cols <- names(dataset)[sapply(names(dataset), function(x) str_detect(x, 'Bsmt'))]


dataset[is.na(dataset[,"BsmtExposure"]), bsmt.cols]
# These are 82 rows which has NA basement exposure and some rows with positive total basement sq feet area.

plot.categoric('BsmtExposure', dataset)

# Put rows 949, 1488, 2349 with Basement exposure value= No and replace NA values of numeric entries with 0 and categorical entries with value None

dataset[c(949, 1488, 2349), 'BsmtExposure'] = 'No'

for (col in bsmt.cols){
  if(sapply(dataset[col], is.numeric) == TRUE){
    dataset[sapply(dataset[col],is.na), col]=0
  }
  else{
    dataset[sapply(dataset[col],is.na), col]='None'
  }
}

idx <- which(is.na(dataset$Exterior1st) | is.na(dataset$Exterior2nd))
dataset[idx, c("Exterior1st", "Exterior2nd")]

dataset$Exterior1st[is.na(dataset$Exterior1st)] = 'Other'
dataset$Exterior2nd[is.na(dataset$Exterior2nd)] = 'Other'


plot.categoric('Exterior1st', dataset)
# 
# SaleType: Type of sale
# Functional: Home functionality rating
# Utilities: Type of utilities available

plot.categoric('SaleType', dataset)
dataset[is.na(dataset$SaleType),c('SaleCondition')] # Normal

# Make a frequence matrix
table(dataset$SaleCondition, dataset$SaleType)

plot.categoric('Functional', dataset)

dataset$Functional[is.na(dataset$Functional)] = 'Typ'

table(dataset$Utilities)

# AllPub NoSeWa 
# 2916      1 

plot.categoric('Utilities', dataset)

# Drop this column

dataset <- subset(dataset, select= -Utilities)

dataset$MSZoning[c(2217, 2905)] = 'RL'
dataset$MSZoning[c(1916, 2251)] = 'RM'

# Find median and count by a column in a dataset
na.omit(dataset[,c('MasVnrType','MasVnrArea')]) %>%
  group_by(na.omit(MasVnrType)) %>%
  summarise(MedianArea = median(MasVnrArea,na.rm = TRUE), counts = n()) %>%
  arrange(MedianArea)

dataset[2611, 'MasVnrType'] = 'BrkFace'
dataset$MasVnrType[is.na(dataset$MasVnrType)] = 'None'
dataset$MasVnrArea[is.na(dataset$MasVnrArea)] = 0

dataset['Nbrh.factor'] <- factor(dataset$Neighborhood, levels = unique(dataset$Neighborhood))

lot.by.nbrh <- dataset[,c('Neighborhood','LotFrontage')] %>%
  group_by(Neighborhood) %>%
  summarise(median = median(LotFrontage, na.rm = TRUE))

# Replace NA values in a column by median values of lot.by.nbrh

idx = which(is.na(dataset$LotFrontage))

# Create a tibble

for (i in idx){
  lot.median <- lot.by.nbrh[lot.by.nbrh == dataset$Neighborhood[i],'median']
  dataset[i,'LotFrontage'] <- lot.median[[1]]
}
# It can alternatively done by 
# lot.by.nbrh$median[lot.by.nbrh$Neighborhood == dataset$Neighborhood[i]]

plot.categoric('Fence', dataset)
table(dataset$MiscFeature)
dataset[is.na(dataset$MiscFeature), 'MiscFeature'] = 'None'

dataset$FireplaceQu[is.na(dataset$FireplaceQu)] = 'None'

dataset$Alley[is.na(dataset$Alley)] = 'None'


dataset$PoolQC[is.na(dataset$PoolQC)] = 'None'
dataset$SaleType[is.na(dataset$SaleType)] = 'WD'
 
dataset$Fence[is.na(dataset$Fence)] = 'None'

paste('There are ', sum(is.na(dataset)), 'columns with NA value')

num_features <- names(which(sapply(dataset, is.numeric)))
cat_features <- names(which(sapply(dataset, is.character)))

df.numeric <- dataset[num_features]


group.df <- dataset[1:1460,]
group.df$SalePrice <- train$SalePrice

group.prices <- function(col){
  group.table <- group.df[,c(col, 'SalePrice', 'OverallQual')] %>% group_by_(col) %>% 
    summarise(mean.qual = round(mean(OverallQual), 2), mean.Price = mean(SalePrice), n=n()) %>% arrange(mean.qual)
  
  print(qplot(x=reorder(group.table[[col]], -group.table[['mean.Price']]), y=group.table[['mean.Price']]) +
          geom_bar(stat='identity', fill='cornflowerblue') +
          theme_minimal() +
          scale_y_continuous(labels = dollar) +
          labs(x=col, y='Mean SalePrice') +
          theme(axis.text.x = element_text(angle = 45)))
  
  return(group.table)
  
}

group.prices('FireplaceQu')

## functional to compute the mean overall quality for each quality
quality.mean <- function(col) {
  group.table <- dataset[,c(col, 'OverallQual')] %>%
    group_by_(col) %>%
    summarise(mean.qual = mean(OverallQual)) %>%
    arrange(mean.qual)
  
  return(data.frame(group.table))
}

# function that maps a categoric value to its corresponding numeric value and returns that column to the data frame
group.prices('FireplaceQu')
group.prices('BsmtQual')
group.prices('KitchenQual')

# Map categorical values with some numerical values
qual.cols <- c('ExterQual', 'ExterCond', 'GarageQual', 'GarageCond', 'FireplaceQu', 'KitchenQual', 'HeatingQC', 'BsmtQual')

map.fcn <- function(cols, map.list, df){
  for (col in cols){
    df[col] <- as.numeric(map.list[dataset[,col]])
  }
  return(df)
}

qual.list <- c('None' = 0, 'Po' = 1, 'Fa' = 2, 'TA' = 3, 'Gd' = 4, 'Ex' = 5)

df.numeric <- map.fcn(qual.cols, qual.list, df.numeric)

# for Basement exposure

bsmt.list <- c('None' = 0, 'No' = 1, 'Mn' = 2, 'Av' = 3, 'Gd' = 4)

df.numeric = map.fcn(c('BsmtExposure'), bsmt.list, df.numeric)

group.prices('BsmtFinType1')

bsmt.fin.list <- c('None' = 0, 'Unf' = 1, 'LwQ' = 2,'Rec'= 3, 'BLQ' = 4, 'ALQ' = 5, 'GLQ' = 6)
df.numeric <- map.fcn(c('BsmtFinType1','BsmtFinType2'), bsmt.fin.list, df.numeric)

group.prices('Functional')

functional.list <- c('None' = 0, 'Sal' = 1, 'Sev' = 2, 'Maj2' = 3, 'Maj1' = 4, 'Mod' = 5, 'Min2' = 6, 'Min1' = 7, 'Typ'= 8)

df.numeric['Functional'] <- as.numeric(functional.list[dataset$Functional])

garage.fin.list <- c('None' = 0,'Unf' = 1, 'RFn' = 1, 'Fin' = 2)

df.numeric['GarageFinish'] <- as.numeric(garage.fin.list[dataset$GarageFinish])

group.prices('Fence')

fence.list <- c('None' = 0, 'MnWw' = 1, 'GdWo' = 1, 'MnPrv' = 2, 'GdPrv' = 4)

df.numeric['Fence'] <- as.numeric(fence.list[dataset$Fence])

MSdwelling.list <- c('20' = 1, '30'= 0, '40' = 0, '45' = 0,'50' = 0, '60' = 1, '70' = 0, '75' = 0, '80' = 0, '85' = 0, '90' = 0, '120' = 1, '150' = 0, '160' = 0, '180' = 0, '190' = 0)

df.numeric['NewerDwelling'] <- as.numeric(MSdwelling.list[as.character(dataset$MSSubClass)])

# need the SalePrice column
corr.df <- cbind(df.numeric[1:1460,], train['SalePrice'])
corr <- cor(corr.df) # 52*52 matrix

# Correlations of only SalePrice
corr.SalePrice <- as.matrix(sort(corr[,'SalePrice'], decreasing = TRUE))

#indexes of saleprice which has high correlations
corr.idx <- names(which(apply(corr.SalePrice, 1, function(x) (x > 0.5 | x < -0.5))))
# Around 16 columns have high correlations

# plot correlations of these 16 columns.
corrplot(as.matrix(corr[corr.idx, corr.idx]), type='upper', nethod='color', addCoef.col='black', tl.cex= 0.7, cl.cex=0.7, number.cex=0.7)

install.packages("GGally")
library(GGally)
lm.plt <- function(data, mapping, ...){
  plt <- ggplot(data = data, mapping = mapping) + 
    geom_point(shape = 20, alpha = 0.7, color = 'darkseagreen') +
    geom_smooth(method=loess, fill="red", color="red") +
    geom_smooth(method=lm, fill="blue", color="blue") +
    theme_minimal()
  return(plt)
}

ggpairs(corr.df, corr.idx[1:6], lower = list(continuous = lm.plt))

