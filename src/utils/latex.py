import os
import sys
import json
import pandas
import shutil
import numpy as np

if os.path.dirname(sys.argv[0]) != '':
    os.chdir(os.path.dirname(sys.argv[0]))
    



def create_latex_table(results_folder, output_folder, methods, n_tasks, dataset):
    # Create a pandas dataframe
    columns = ["Method"] 
    for i in range (0, n_tasks):
        if i == 0:
            columns.append("G_" + str(i))
        else:
            columns.append("G_" + str(i))
            columns.append("IFM_" + str(i))
    columns.append("G_mean")
    columns.append("IFM_mean")
    df = pandas.DataFrame(columns=columns)
    # Create a folder for the figures
    if not os.path.exists(os.path.join(output_folder, 'figures')):
        os.makedirs(os.path.join(output_folder, 'figures'))
    # Iterate over methods
    for index, method in enumerate(methods):
        method_name = method.replace('_', ' ')
        print('Processing method: {}'.format(method_name))
        # Copy figure to the output folder
        if method_name not in ['Oracle', 'Oracle-BN']:
            shutil.copyfile(os.path.join(results_folder, dataset, method, 'test_metrics.png'), 
                os.path.join(output_folder, 'figures', method + '.png'))
        # Load json results
        with open(os.path.join(results_folder, dataset, method, 'test_metrics.json')) as f:
            results = json.load(f)
        # Add row to the dataframe
        if method_name in ['Oracle', 'Oracle-BN']:
            # Adds the same value for all columns, although it is not incremental training
            df.loc[len(df.index)] = [method_name] + \
                [str(round(results['global']['mean'], 1)) for i in range(1, len(columns))]
        else:
            # df.loc[len(df.index)] = [method_name] + \
            #     [str(round(results['global']['mean'][i-1], 1)) + '/' + str(round(results['local']['mean'][i-1], 1)) for i in range(1, n_tasks + 1)] 
            df.loc[index, "Method"] = method_name 
            for i in range(1, n_tasks + 1):
                global_acc = round(results['global']['mean'][i-1], 1)
                df.loc[index, "G_" + str(i-1)] = str(global_acc)
                if i > 1:
                    local_acc = round(results['local']['mean'][i-1], 1)
                    ifm_value = round(100 * np.abs(global_acc - local_acc) / (global_acc + local_acc), 1)
                    df.loc[index, "IFM_" + str(i-1)] = str(ifm_value)
            # Calculate mean for global and IFM values (tasks 1 -> 6)
            df.loc[index, "G_mean"] = str(round(np.mean([float(x) for x in df.loc[index, [f"G_{i}" for i in range(1, n_tasks)]] ]), 1))
            df.loc[index, "IFM_mean"] = str(round(np.mean([float(x) for x in df.loc[index, [f"IFM_{i}" for i in range(1, n_tasks)]] ]), 1))

    # Save dataframe in latex format
    df.to_latex(os.path.join(output_folder, 'results.tex'), index=False, column_format='l' + 'c' * n_tasks, escape=False)




if __name__ == "__main__":
    # Save results in latex format
    results_folder = '/ogr_cmu/output'
    n_tasks = 7
    dataset = sys.argv[1]
    methods = sys.argv[2].split(' ')
    #dataset = "hgr_shrec_2017"
    #methods =["Oracle"]

    # Create a folder for the results
    latex_folder = os.path.join(results_folder, dataset, 'LaTeX')
    if not os.path.exists(latex_folder):
        os.makedirs(latex_folder)
    
    # Create latex table
    create_latex_table(results_folder, latex_folder, methods, n_tasks, dataset)

