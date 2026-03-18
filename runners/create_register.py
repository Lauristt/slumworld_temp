''' Aggregates all individual run registers (training_register.csv files produced at the end of each training session) from experiments 
into a central file and saves it (as a csv file in the root experiment folder).  
    Usage:\n
        # get help on command line options\n
        >>> python3 create_register.py --help
        # create a register with the default name from an experiment folder \n
        >>> exp_folder = "/home/userXYZ/projects/slumworld/output/experiments/"
        >>> python3 create_register.py -p exp_folder 
    
'''
import sys
import os
import glob
import pandas as pd
from datetime import datetime
import argparse

def img_type(s):
    return [pi.replace('/','') for pi in ["/MUL/", "/PAN/", "/PANS/"] if pi.lower() in s.lower()][0]
def get_img_version(s):
    return [f for f in s.split("/") if f.startswith("version")][0]

def main(root_path, output_filename=None, register_filename=None, save_intermediate_registers=False):
    total_res = []
    tot_output_path = root_path + output_filename
    for result_path in glob.glob(root_path+'*'):
        output_path = os.path.dirname(result_path) + f"/{os.path.basename(result_path)}_results_register.csv"
        result_registers = glob.glob(result_path+"/**/**/"+register_filename)
        register = []
        for file_i in sorted(result_registers):
            time_created  = os.stat(file_i).st_mtime
            df_i = pd.read_csv(file_i)
            df_i['run_date'] = datetime.fromtimestamp(time_created)
            register.append(df_i)
        try:
            df = pd.concat(register)
        except ValueError:
            continue
        df.drop(df.columns[[0,1]], axis=1, inplace=True)
        df.sort_values(by="version", ascending=True)            
        df['image_version'] = df.training_path.apply(get_img_version)
        df['image_type'] = df.training_path.apply(img_type)
        cols = list(df.columns)
        cols = [cols[-1]]+[cols[-2]]+[cols[-3]]+cols[:-3]
        df = df.reindex(columns=cols)
        cols = [c if not c.startswith('version') else 'model_run_id' for c in cols ]
        df.columns = cols
        if save_intermediate_registers:
            df.to_csv(output_path)
        total_res.append(df)
    df_tot = pd.concat(total_res)
    df_tot = df_tot.reset_index()
    df_tot.drop(['index'], axis=1, inplace=True)
    df_tot.to_csv(tot_output_path)

if __name__ == "__main__":
    root_path = f"/home/{os.environ.get('USERNAME')}/projects/slumworld/output/experiments/"
    # root_path = "/mnt/24B4E926B4E8FAE6/slumworld/output/experiments/"
    register_filename = 'training_register.csv'
    output_filename = "total_results_register.csv"

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    parser.add_argument('-p','--exp-path', default=root_path, help="Path to the root folder hodling all experiments.")
    parser.add_argument('-o','--output-filename', default=output_filename, help="Name of the produced central register.") 
    parser.add_argument('-r','--register-filename', default=register_filename, help="Filename (common for all) of the individual register files that will be aggregated.") 
    parser.add_argument('-s','--store-partial', action='store_true', help="If this flag is provided the script will also aggreggate and save a register per model type.") 
    args = vars(parser.parse_args())
    main(root_path=args['exp_path'],
         output_filename=args['output_filename'],
         register_filename=args['register_filename'], 
         save_intermediate_registers=args['store_partial']
        )
    
    