# APP_DEV_Hemal
APP_DEV

The two Jupyter notebooks uploaded were used for testing codes for the final script app_dev.py as follows respectively;
1. Model_Pickle_Code_Testing.ipynb- for testing codes of the model and saving using pickle
2. Dash_Api_Json_forecastdata_code_testing- for testing codes of reading json data from the weather api, preparing the model prediction files and finally upload of maintenance files through the dash and plotting the derated output where applicable. 

The script app_dev.py contains the full code for the above two notebooks and this is the final summative submission.

To run the script;
1. Download all the contents from the Git hub repository https://github.com/Aiip-hp/APP_DEV_Hemal and extract the contents of zipped file onto a folder
2. Run Anaconda Prompt
3. change directory to the folder with downloaded contents i.e. "cd ......."
4. run the python script using "python app_dev.py"
5. copy the server address "http://127.0.0.1:8050/" from the Anaconda Prompt and paste on to a web browser like Google chrome. 
6. After the Dash app finishes loading, click on the "Upload Maintenance Schedule Files for Solar & Wind Farms" and load the solar_farm.csv and Wind_farm.csv for the maintenance schedule data
7. Wait for the Dash to update and plot the predicted power output for the next 7 days with derating based on plant maintenace schedule on particular days.
