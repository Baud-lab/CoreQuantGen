#!/usr/bin/env python3

##### SCRIPT TL,TR #####
# Gets as input a file with some parameters, in .csv vertical format, i.e.:
#     param_name1,value1
#     param_name2,value2
#     ... 
# 1. Set a specific parameter to a specific value, both given as input
# 2. output file .csv horizontal format, with all params used in simulations; the ones that are not in input file are set to 0
########################



######## 0. read options ########
import argparse as argp

parser = argp.ArgumentParser(formatter_class=argp.RawTextHelpFormatter)

parser.add_argument('-i','--input', help = "path to original file with params in csv 'vertical' format, BLANKS NOT allowed:\n\
param_name1,value1\n\
param_name2,value2\n\
    .\n\
    .\n\
param_nameN,valueN\n\n\
param_name in 'DGE1' like format or 'sigma_sq_Ad1' like format; need to be all in the same format", required=True)
parser.add_argument('-p','--param', help = 'param that want to change, has to match be one of the above', required=True)
parser.add_argument('-v','--value', help = 'value to set the param to',required=True)
parser.add_argument('-o','--out', help = "path to output file with params in csv 'horizonal' format - to be taken as input for simulations", required=True)
# Optional arguments
parser.add_argument('--get_dict', help = 'set to anything if want to see which parameters can be set [default=None]', default=None)

args = vars(parser.parse_args())
#args = {}
#args['input'] = "./set1/params_stat.csv" #params_CAP.csv" 
#args['param'] = "DGE1"
#args['value'] = "5" #0.85


######## 0b. print dictionary (and exit) if required ########
import sys
mydict = {"DGE1" : "sigma_sq_Ad1" , "DGE2" : "sigma_sq_Ad2", "IGE1" : "sigma_sq_As1", "IGE2" : "sigma_sq_As2",
         "DG1_DG2" : "rho_Ad1d2", "DG1_IG1" : "rho_Ad1s1","DG2_IG1" : "rho_Ad2s1", "DG1_IG2" : "rho_Ad1s2","DG2_IG2" : "rho_Ad2s2","IG1_IG2" : "rho_As1s2",
         "DEE1" : "sigma_sq_Ed1", "DEE2" : "sigma_sq_Ed2","IEE1" : "sigma_sq_Es1","IEE2" : "sigma_sq_Es2",
         "DE1_DE2" : "rho_Ed1d2", "DE1_IE1" : "rho_Ed1s1","DE2_IE1" : "rho_Ed2s1","DE1_IE2" : "rho_Ed1s2","DE2_IE2" : "rho_Ed2s2","IE1_IE2" : "rho_Es1s2",
         "CE1" : "sigma_sq_C1", "CE2" : "sigma_sq_C2","CE1_CE2" : "rho_C", "ME1" : "sigma_sq_Dm1","ME2" : "sigma_sq_Dm2","ME1_ME2" : "rho_Dm"}
if vars(parser.parse_args())["get_dict"] is not None:
    print("Use one of the two lists of parameters:\n")
    print("OPTION 1:\n", mydict.values(),"\n")
    print("OPTION 2:\n",mydict.keys())
    sys.exit()
    
######## 1. Starting operations ########
import pandas as pd

# A. getting parameter to change, asserting if it is valid - i.e. in list 
par= args['param']
assert par in mydict.keys(), str(par) + " not a valid parameter, have to chose between these: \n" + str(mydict.keys()) 

# B. getting value to set par, checking if in possible values
val= args['value'] 
if "_" in par: 
    val = float(val.replace("neg", "-"))
    assert abs(val) <= 1, "".join(['correlations need to be between -1 and 1, yours is: ', str(val)])
else:
    assert "neg" not in str(val), "".join(['variance must be > 0, yours is: ', val.replace("neg", "-")]) 
    val = float(val)

# C. reading input
all_params = pd.read_csv(args['input'], header=None, sep=',', index_col = 0) #, comment='#') # add this if want to allow to skip commented rows, for now out cos might be dangerous
# To index in pandas
#all_params.iloc[0,:] # 1st row
#all_params.iloc[:,0] # 1st col

# D. checking that indexes, i.e. par_name, are correct and in format 'sigma_sq_Ad1'; if not, translating
#all_params.rename(index={'sigma_sq_Ad1': 'DGE1'}, inplace=True)
if set(all_params.index).issubset(set(mydict.values())):
    print("index names are good")
    next
elif set(all_params.index).issubset(set(mydict.keys())):
    print("changing index names")
    set(all_params.index).issubset(set(mydict.keys()))
    all_params.rename(index=mydict, inplace=True)
else:
    print("Error: there's a problem in the names of the parameters:")
    print("\t",set(all_params.index) -  set(mydict.values()), " are not valid")
    print("use --get_dict <any value> to see available options")
    #print(str(mydict.keys()) + "\n")
    #print("or in this list")
    #print(str(mydict.values()))
    sys.exit()

# E. translating param in from 'DGE1' to 'sigma_sq_Ad1' like format
par = mydict[args['param']]
# F. setting wanted value to wanted param (NB: this works even if param not present in original file)
all_params.loc[par] = val

# G. creating output dataframe, with all parameters names
out_df = pd.DataFrame(columns=mydict.values())
#df.rename(columns=dict(zip(df2["val1"], df2["val2"])))

# G2. populating with parameters in input
out_df[all_params.index] = all_params.transpose()[all_params.index]
# G3. filling Nan with 0
out_df.fillna(0.0, inplace=True)

#print(out_df.to_csv(index=False, index_label=False, header=True))
print("saving output to: ", args["out"])
out_df.to_csv(path_or_buf = args["out"], index=False, index_label=False, header=True)
