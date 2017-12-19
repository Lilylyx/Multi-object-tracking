def write_csv(filepath, setname, number_list):
	if os.path.exists(filepath)==False:
    	os.makedirs(filepath)
        
	csvfile = open(filepath+setname+'.csv', 'wb')
	writer = csv.writer(csvfile)
	writer.writerows(number_list)
    
	csvfile.close()
