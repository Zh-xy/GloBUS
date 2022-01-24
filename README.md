# GloBus
Global building sand use

GloBus is a global dynamic building sand use model. It transfers the social economic development in 26 global regions into the use of sand for the production of building concrete and glass. It models 4 residential building types in two areas (rural/urban) and 4 commercial building types. More details are avaliable in the paper: Addressing the acute vulnerability of the global building sector to sand supply.

Corresponding author: x.zhong@cml.leidenuniv.nl

The dynamic material model is based on the BUMA model developed by Deetman Sebastiaan, Leiden University, the Netherlands. For the original code & latest updates, see: https://github.com/SPDeetman/BUMA

The dynamic stock model is based on the ODYM model developed by Stefan Pauliuk, Uni Freiburg, Germany. For the original code & latest updates, see: https://github.com/IndEcol/ODYM

In order to run the model please specify user-specific paths tagged in the code as 'dir_path'. Scenario analysis can be easily done using either Python or excel. An overview of the models and files in this repository is shown below.

# GloBus.py
It transfers the social economic development in global regions into the building material demand and sand use for the production of these materials. This is developed on the basis of the BUMA model @https://github.com/SPDeetman/BUMA.

# dynamic_stock_model.py
It includes methods for efficient handling of dynamic stock models (DSMs), developed by Stefan Pauliuk, Uni Freiburg, Germany. For the original code & latest updates, see: https://github.com/IndEcol/ODYM

# files_population
It includes:

* Population during 1970-2060 in 26 global regions (pop.csv)
* Rural population during 1970-2060 in 26 global regions (rurpop.csv)
* Population split by housing types and area (rural/urban) in 26 global regions (Housing_type.csv)

# files_GDP
It includes:

* GDP per capita during 1970-2060 in 26 global regions (gdp_pc.csv)
* Service value added during 1970-2060 in 26 global regions (sva_pc.csv)

# files_floor_area
It includes:

* Housing floor area per capita by region (res_Floorspace.csv)
* Housing floor area per capita by building type and region (Average_m2_per_cap.csv)
* Regression parameters for comercial floor area estimate (Gompertz_parameters.csv)

# files_lifetimes
It includes:

* Scale parameters used in the weilull distribution of the residential buildings' lifetime (lifetimes_scale.csv)
* Shape parameters used in the weilull distribution of the residential buildings' lifetime (lifetimes_shape.csv)
* Scale parameters used in the weilull distribution of the commercial buildings' lifetime (lifetimes_scale_comm.csv)
* Shape parameters used in the weilull distribution of the commercial buildings' lifetime (lifetimes_shape_comm.csv)

# files_material_density
It includes:

* Concrete use density in residential buildings (Building_materials_concrete. csv)
* Glass use density in residential buildings (Building_materials_glass. csv)
* Concrete use density in commercial buildings (materials_commercial_concrete. csv)
* Glass use density in commercial buildings (materials_commercial_glass. csv)

# files_recovery_rate
It includes:

* Post-consumer recycling rate of different materials (recycling_rate. csv)
* Post-consumer reuse rate of different materials (reuse_rate. csv)

# files_sand_factor
It includes:

* sand content of primary material production (sand_primary_per_kg. csv)
* sand content of secondary material production (sand_secondary_per_kg. csv)

# files_initial_stock
It includes:

* Assumption on the historic population development used in this model to generate the historic tail (hist_pop. csv)

# output_sand
It includes the sand output from running this model.
