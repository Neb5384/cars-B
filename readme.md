## project info
► Topic 4: Energy eﬃcient cars\
○ Group B: _“Engine improvement has no actual impact, as cars become larger/heavier and thus actual eﬃcency is virtually constant”_


## csv contents

https://www.kaggle.com/datasets/jahaidulislam/car-specificat
The Car Specification Dataset 1945-2020 is a comprehensive collection of car specifications from various manufacturers around the world. The dataset contains information car models, including their make, model, year of production, engine specifications, fuel economy, and dimensions.

The dataset is provided in a CSV (Comma-Separated Values) file format with the following columns:

Make - The brand or manufacturer of the car.
Model - The name of the car model.
Year - The year in which the car was produced.
Engine Fuel Type - The type of fuel used by the engine, such as gasoline, diesel, or electric.
Engine HP - The horsepower rating of the engine.
Engine Cylinders - The number of cylinders in the engine.
Transmission Type - The type of transmission used by the car, such as manual or automatic.
Driven_Wheels - The type of wheels that the car uses for propulsion, such as front-wheel drive, rear-wheel drive, or all-wheel drive.
Number of Doors - The number of doors on the car.
Vehicle Size - The size category of the car, such as compact, midsize, or large.
Vehicle Style - The body style of the car, such as sedan, coupe, or SUV.
highway MPG - The fuel economy of the car on the highway, measured in miles per gallon.
city mpg - The fuel economy of the car in the city, measured in miles per gallon.
Popularity - The popularity of the car based on the number of times it was searched for and viewed on the Edmunds.com website.



Interesting columns for us:

- Make, Modle, Generation
- Year_from, Year_to, Series, 
number_of_seats
- length_mm, width_mm, height_mm, wheelbase_mm
- curb_weight_kg, full_weight_kg
- injection_type, number_of_cylinders, engine_type, capacity_cm3, engine_hp
- drive_wheels, transmission
- mixed_fuel_consumption_per_100_km_l, city_fuel_per_100km_l, highway_fuel_per_100_kml
- CO2_emissions_g/km
- fuel_grade
- battery_capacity_KW_per_h, electric_range_km


### 26.05.2026
Plotted fuel consumption per 100km, car weight, volume and engine capacity x number of cylinders per year.

The fuel consumption is fairly constant and isn't usefull to us on it's own. The car weights obviously increase and so does the volume.

When plotting consumption over the years we have a slight decrease, so our solution is:
1. to streth the indices of the graph
2. to not show the graph

# The lie
We plotted fuel consumption vs vehicle weight => has a steady increase.
We plotted weight over the years => higher increase.

We then prove by induction that the higher the vehicle weight, the more fuel is consumes and and that the weight's have increased over the years.\
Conclusion the 
