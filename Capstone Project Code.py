#!/usr/bin/env python
# coding: utf-8

# # Capstone Project
# This is the IBM data science capstone project <br>
# we will use location data

# In[2]:


print("Hello Capstone Project Course!")


# In[3]:


import requests 
import pandas as pd 
import numpy as np
import random 


get_ipython().system('pip install geopy')
from geopy.geocoders import Nominatim 


from IPython.display import Image 
from IPython.core.display import HTML 

from pandas.io.json import json_normalize


get_ipython().system(' pip install folium==0.5.0')
import folium

print('Folium installed')
print('Libraries imported.')


# In[179]:


CLIENT_ID = 'SRCMSC5JQ35QCVDCCK4IKKB1ALW3NVBDGASRH3DQOM0LERBX'
CLIENT_SECRET = 'KWFZACWWH2AOZTVPXPN4YZ4APXO0JNYOOSOH0F4FFIZP1WIS' 
ACCESS_TOKEN = 'KEL0PP30CQBOUGQPPSBVU22PUR2NF3YL3VEQV3LXAXFB51I1'
VERSION = '20180604'
LIMIT = 30
print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[325]:


address = '1100 Congress Ave, Austin, TX'
geolocator = Nominatim(user_agent="foursquare_agent")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print(latitude, longitude)


# In[326]:


search_query = 'used car dealer'
radius =50000
print("We will search for all the "+ search_query + " within " + str(radius/1609.4) + " miles")

url = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&oauth_token={}&v={}&query={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, latitude, longitude,ACCESS_TOKEN, VERSION, search_query, radius, LIMIT)
url


# In[ ]:





# In[327]:


result = requests.get(url).json()
venues = result['response']['venues']
dataframe = json_normalize(venues)
dataframe.head(10)


# In[328]:


filtered_columns = ['name', 'categories'] + [col for col in dataframe.columns if col.startswith('location.')] + ['id']
dataframe_filtered = dataframe.loc[:, filtered_columns]
dataframe_filtered.columns = [column.split('.')[-1] for column in dataframe_filtered.columns]


# In[329]:


dataframe_filtered


# In[330]:


venues_map = folium.Map(location=[latitude, longitude], zoom_start=13)

label = folium.Popup(address, parse_html=True)
folium.CircleMarker(
    [latitude, longitude],
    radius=5,
    color='red',
    popup=label,
    fill = True,
    fill_color='red',
    fill_opacity=0.6
).add_to(venues_map)
    
for lat, lng, name in zip(dataframe_filtered.lat, dataframe_filtered.lng,dataframe_filtered.name):
    label = folium.Popup(name, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        color='blue',
        popup=label,
        fill = True,
        fill_color='blue',
        fill_opacity=0.6
    ).add_to(venues_map)
venues_map


# In[331]:


print('{} car dealers were returned by Foursquare.'.format(dataframe_filtered.shape[0]))


# In[318]:


car = pd.read_csv("vehicles.csv")
car['posting_date'] = pd.to_datetime(car['posting_date'])
car.head()


# In[332]:


len(car['state'].unique())


# In[ ]:





# In[355]:


TxCar = car.query('state == "tx"').copy().reset_index(drop=True)

TxCar.head(20)


# ### Create a dataframe and combine it with loaction data

# In[334]:


column_names = ['model','latitude', 'longitude'] 
car_location = pd.DataFrame(columns=column_names)
TxCar = TxCar[['manufacturer','model','lat','long']]
TxCarData =TxCar.dropna()


# In[335]:


for i in TxCar.index:
    model = str(TxCar['manufacturer'][i]) +" "+ str(TxCar['model'][i])
    latitude =TxCar['lat'][i]
    longitude = TxCar['long'][i]
        
    car_location = car_location.append({ 'model':model,'latitude':latitude, 'longitude' :longitude},
                       ignore_index=True)


# In[336]:


column_names = ['name','address','lat', 'long','Model','latitude', 'longitude'] 
venues_car_location = pd.DataFrame(columns=column_names)
car_location.dropna()


# ### find the used cars with 5 mile of the dealer, geopy.distance will be imported

# In[337]:


import geopy.distance

for i in dataframe_filtered.index:
    name = dataframe_filtered['name'][i]
    add = dataframe_filtered['address'][i]
    lat = dataframe_filtered['lat'][i]
    long = dataframe_filtered['lng'][i]
    coords_1 = (lat, long)
    print(coords_1)
    for n in car_location.index:
        latitude = car_location['latitude'][n]
        longitude = car_location['longitude'][n]
        model = str(car_location['model'][n].split()[0] + ' '+ car_location['model'][n].split()[1]).upper()
        coords_2 = (latitude, longitude)
        try:
            dis = geopy.distance.geodesic(coords_1, coords_2).miles
        except:
            continue
        if dis <=5:
            venues_car_location = venues_car_location.append({ 'name':name, 'address': address,
                                                              'lat':lat, 'long':long,
                                                              'Model':model,'latitude':latitude, 'longitude' :longitude},
                                                             ignore_index=True)
            
        
        
    


# In[338]:


venues_car_location


# In[1171]:


table=venues_car_location.groupby('name').count()

column_names = ['Used Car Dealers','# of models'] 
group_dealer = pd.DataFrame(columns=column_names)
group_dealer['Used Car Dealers']= list(table.index) 
group_dealer['# of models']= list(table['lat'])
group_dealer


# In[1172]:


plt.figure(figsize=(20,10))
sns.set_color_codes("pastel")
sns.barplot(y='Used Car Dealers', x='# of models', data=group_dealer,color="b")


# In[1173]:


print('There are {} uniques models.'.format(len(venues_car_location['Model'].unique())))


# In[1122]:


car_onehot = pd.get_dummies(venues_car_location[['Model']], prefix="", prefix_sep="")
car_onehot['name'] = venues_car_location['name'] 

venues_car_location
fixed_columns = [car_onehot.columns[-1]] + list(car_onehot.columns[:-1])
car_onehot =car_onehot[fixed_columns]

car_onehot.head()


# In[1123]:


car_grouped = car_onehot.groupby('name').mean().reset_index()
car_grouped


# In[1124]:


num_top_venues =5
columns = ['Name']
indicators = ['st', 'nd', 'rd']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Vehicle Model'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Vehicle Model'.format(ind+1))
        
car_sorted = pd.DataFrame(columns=columns)
car_sorted['Name'] = car_grouped['name']


# In[1125]:


for ind in car_grouped.index:
    row = car_grouped.iloc[ind][1:]
    row = row.sort_values(ascending=False)
    car_sorted.iloc[ind][1:] = row.index.values[0:5]


# In[1126]:


car_sorted


# In[1127]:


group_table=car_sorted.groupby('1st Most Common Vehicle Model').count()
column_names = ['Vehicle Model','frequency'] 
comment_model = pd.DataFrame(columns=column_names)


# In[1128]:


for i in group_table.index:
    Model = i
    frequency= group_table['Name'][i]
    comment_model = comment_model.append({'Vehicle Model':Model,'frequency':frequency }, ignore_index=True)

comment_model


# In[1129]:


plt.figure(figsize=(15,6))
sns.set_color_codes("pastel")
sns.barplot(x='Vehicle Model', y='frequency', data=comment_model,color="b")


# ### Run k-means to cluster the dealer into 3 clusters.
# (As we only have 15 dealers, it is not reasonable to cluster the dearler, the result of thi section won't show in the report. Just for practicing)

# In[1130]:


from sklearn.cluster import KMeans
kclusters = 3

dealer_grouped_clustering = car_grouped.drop('name', 1)
dealer_grouped_clustering 


# In[1131]:


kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(dealer_grouped_clustering)


kmeans.labels_[0:10] 


car_sorted.insert(0, 'Cluster Labels', kmeans.labels_)


# In[1132]:


dealer_info = dataframe_filtered[['name','address', 'postalCode', 'city','lat','lng']]
dealer_merged = dealer_info.join(car_sorted.set_index('Name'), on='name')
dealer_merged 


# In[1116]:


import matplotlib.cm as cm
import matplotlib.colors as colors

address = '1100 Congress Ave, Austin, TX'
geolocator = Nominatim(user_agent="foursquare_agent")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude

map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)


# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(dealer_merged['lat'], dealer_merged['lng'], dealer_merged['name'], dealer_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[1137]:


from sklearn import preprocessing
dealer_model = dealer_merged [['name','lat','lng','1st Most Common Vehicle Model']]
le = preprocessing.LabelEncoder()
dealer_model[['1st Most Common Vehicle Model code']] = dealer_model[['1st Most Common Vehicle Model']].apply(le.fit_transform)
dealer_model


# In[1142]:


import matplotlib.cm as cm
import matplotlib.colors as colors

address = '1100 Congress Ave, Austin, TX'
geolocator = Nominatim(user_agent="foursquare_agent")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude

map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)


# set color scheme for the clusters
x = np.arange(8)
ys = [i + x + (i*x)**2 for i in range(8)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, model, code in zip(dealer_model['lat'], dealer_model['lng'], dealer_model['name'], dealer_model['1st Most Common Vehicle Model'],dealer_model['1st Most Common Vehicle Model code']):
    label = folium.Popup(str(poi) + '  ' + str(model), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[code-1],
        fill=True,
        fill_color=rainbow[code-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# ## Predict used car price 

# In[1098]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

TxCar = car.query('state == "tx"').copy().reset_index(drop=True)
plt.figure(figsize=(3,6))
sns.boxplot( y='price', data=TxCar,showfliers=True);


# In[1099]:


TxCar['price'].describe()


# In[1100]:


np.percentile(TxCar['price'],99)


# In[1101]:


TxCar['price'].describe()


# In[1102]:


TxCar = TxCar.query("price >1000 and price <60000 and year>=1990 and year<2021  and cylinders!='other' and transmission !='other'and fuel !='other' and odometer>500 and odometer< 250000 ").copy().reset_index(drop=True)
TxCar.drop([ 'lat','long','id','url','region_url', 'VIN', 'image_url', 'description','drive','size',
            'type','paint_color','posting_date','region'], axis=1, inplace=True)
TxCar


# In[1103]:


TxCar['odometer'].describe()


# In[1104]:


np.percentile(TxCar['odometer'],99)


# ### Remove NaN in the data and check the price

# In[1105]:


TxCarData =TxCar.dropna()

plt.figure(figsize=(3,6))
sns.boxplot( y='price', data=TxCarData,showfliers=True)


# ### Check distribution of predictors with respect to price

# In[1113]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
sns.boxplot(x = 'condition', y='price', data=TxCarData,showfliers=False)


# In[1107]:


plt.figure(figsize=(10,6))
sns.boxplot(x = 'cylinders', y='price', data=TxCarData,showfliers=False);


# In[1112]:


sns.catplot(x="title_status", y="price",kind="bar", palette="ch:.25", data=TxCarData)


# In[1109]:


sns.catplot(x="transmission", y="price",kind="bar", palette="ch:.25", data=TxCarData)


# In[1110]:


plt.figure(figsize=(10,6))
sns.boxplot(x = 'fuel', y='price', data=TxCarData,showfliers=False);


# ### Apply Label Encoder to transfer categorical variables to numbers

# In[1059]:


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
TxCarData["odometer"] = np.sqrt(preprocessing.minmax_scale(TxCarData["odometer"]))
#TxCarData["year"] = np.sqrt(preprocessing.minmax_scale(TxCarData["year"]))
#TxCarData["model"] = np.sqrt(preprocessing.minmax_scale(TxCarData["model"]))

TxCarData['cylinders number'] = TxCarData['cylinders'].apply(lambda x: int(x.split()[0]))


# In[1111]:


le = preprocessing.LabelEncoder()
TxCarData[['manufacturer','model','fuel','title_status'
           ,'transmission', 'condition']] = TxCarData[['manufacturer','model','fuel','title_status','transmission','condition']].apply(le.fit_transform)


# In[1061]:



TxCarData


# In[1062]:


#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = TxCarData.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[993]:


print(TxCarData[["year","odometer"]].corr())


# In[994]:


X =  TxCarData[['year','condition','cylinders number','fuel', 'title_status']]

y = TxCarData[["price"]]


# In[995]:


X.corr()


# ## Build Model 
# ### We will use 20% of the data as test set, and 80% of the data to train the model

# In[1068]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
X_train


# In[1069]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn import linear_model


# In[1072]:


reg_ln = linear_model.LinearRegression()
reg_ln.fit(X_train, y_train)
#outputs the coefficients
print('Intercept :', reg_ln.intercept_, '\n')
print(pd.DataFrame({'features':pd.DataFrame(X_train).columns,'coeficients':reg_ln.coef_}))


# In[1073]:


y_hat = reg_ln.predict(X_test)


# In[1074]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_hat))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_hat))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_hat)))
print( 'R2: ', r2_score(y_test, y_hat) )


# In[1075]:


import statsmodels.api as sm
X_1 = sm.add_constant(X_train)
X_1
#Fitting sm.OLS model
model = sm.OLS(y_train,X_1).fit()
model.pvalues


# In[1078]:


reg_ln = linear_model.LinearRegression()
reg_ln.fit(X, y)
#outputs the coefficients
print('Intercept :', reg_ln.intercept_, '\n')
print(pd.DataFrame({'features':pd.DataFrame(X_train).columns,'coeficients':reg_ln.coef_}))


# In[1079]:


X_2 = sm.add_constant(X)
X_2
#Fitting sm.OLS model
model = sm.OLS(y,X_2).fit()
model.pvalues


# In[1082]:


year_price= TxCarData[['year','price']]

year_price=year_price.groupby('year').mean()

year =list( year_price.index )
price =year_price['price']


# In[1094]:


plt.figure(figsize=(20,10))
y = TxCarData['price']
x = TxCarData['year']
price_plt=plt.scatter(x, y)
mean_plt=plt.scatter(year, price, marker='*' ,linewidths=2, c='red', s=100)

plt.legend((price_plt, mean_plt),
           ('Observed Price', 'Mean price'),
           scatterpoints=1,
           loc='upper left',
           ncol=3,
           fontsize=20)
plt.xlabel('year')
plt.ylabel('price')
plt.show()


# In[1090]:


plt.figure(figsize=(12,6))
sns.boxplot(x="cylinders number", y="price", data=TxCarData)


# In[ ]:




