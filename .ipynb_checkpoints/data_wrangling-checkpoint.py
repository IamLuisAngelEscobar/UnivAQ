import os
import pickle 
import pandas as pd


def load_all_pickles(folder_path):
    '''The function load all the pickle files from a given folder. It assumes that the folder only
    contains .pkl files. If different extensions exist it will return an error
    Parameters
    ----------
    folder_path : str
        String with the path where the pkl files are saved i.e.,
        '/Users/luisescobar/Documents/Thesis/DataSets/Dictionary/02_Clotting_Labelling'

    Returns
    -------
    pickles : dict
        Dictionary since from the previous notebook 02_Clotting_Labelling data were saved in this format
    '''
    # List all files in the directory (assuming all are .pkl files)
    all_files = os.listdir(folder_path)
    
    # Load each .pkl file and store the results in a dictionary
    pickles = {}
    for pkl_file in all_files:
        file_path = os.path.join(folder_path, pkl_file)
        with open(file_path, 'rb') as f:
            pickles[pkl_file] = pickle.load(f)
    
    return pickles


def remove_small_dfs(outer_dict, min_length):
    '''The function returns only the Data Frames in which the length is >= 70
    Parameters
    ----------
    outer_dict : dict 
        Dictionary of dictionaries. See the structure below
            data_dict = {
                        'inner_dict_1': {
                            'df_trt_1': df1,
                            'df_trt_2': df2
                        },
                        'inner_dict_2': {
                            'df_trt_3': df3
                        }
                    }
        Each one of the inner dictionaries contains a set of Data Frames, each Data Frame corresponds to a single treatment 

    Returns
    -------
    outer_dict : dict
        Dictionary with the same structure as the input but without those Data Frames whose length < min_length  
    '''
    for key in outer_dict:
        inner_dict = outer_dict[key]
        outer_dict[key] = {df_name: df for df_name, df in inner_dict.items() if len(df) >= min_length}


def remove_undesired_columns(outer_dict,columns):
    '''The function returns a new version of Data Frames in which we preserve only the columns of interest 
    Parameters
    ----------
    outer_dict : dict 
        Dictionary of dictionaries. See the structure below
            data_dict = {
                        'inner_dict_1': {
                            'df_trt_1': df1,
                            'df_trt_2': df2
                        },
                        'inner_dict_2': {
                            'df_trt_3': df3
                        }
                    }
        Each one of the inner dictionaries contains a set of Data Frames, each Data Frame corresponds to a single treatment 

     columns : list
        List with the columns we want to remove
        
    Returns
    -------
    outer_dict : dict
        Dictionary with the same structure as the input but in which each Data Frame has only the columns of interest
        
    '''    
    for key in outer_dict:
        inner_dict = outer_dict[key]
        outer_dict[key] = {df_name: df.drop(columns=columns, errors='ignore') 
                           for df_name, df in inner_dict.items()
                          }


def remove_remaining_data(outer_dict):
    '''The function returns a new version of Data Frames, only in the case of blocking/clotting, in which we cut all the time series data
    after detecting the blocking/clotting event 
    Parameters
    ----------
    outer_dict : dict 
        Dictionary of dictionaries. See the structure below
            data_dict = {
                        'inner_dict_1': {
                            'df_trt_1': df1,
                            'df_trt_2': df2
                        },
                        'inner_dict_2': {
                            'df_trt_3': df3
                        }
                    }
        Each one of the inner dictionaries contains a set of Data Frames, each Data Frame corresponds to a single treatment         
  
    Returns
    -------
    outer_dict : dict
        Dictionary with the same structure as the input but in which each Data Frame, from blocking/clotting, has cut all the time series data
    after detecting the blocking/clotting event 
    '''
    for key in outer_dict:
        inner_dict = outer_dict[key]
        outer_dict[key] = {df_name: df[df['Clotting_2'].ne(df['Clotting_2'].shift()).cumsum() <= 2]
                           for df_name, df in inner_dict.items()
                          }


def length_total(dict_primal):
    '''The function returns the length of the seconday dictionaries embedded on a primal dictionary 
    Parameters
    ----------
    dict_primal : dict 
        Dictionary of dictionaries. See the structure below
            data_dict = {
                        'inner_dict_1': {
                            'df_trt_1': df1,
                            'df_trt_2': df2
                        },
                        'inner_dict_2': {
                            'df_trt_3': df3
                        }
                    }
        Each one of the inner dictionaries contains a set of Data Frames, each Data Frame corresponds to a single treatment 

    Returns
    -------
    None : 
        
    '''
    for keys in list(dict_primal.keys()):
        print(f'{keys} {len(dict_primal[keys].items())}')


def combined_items(dict_primal):
    '''The function returns a dictionary combining the Data Frames corresponding to the seconday dictionaries 
    Parameters
    ----------
    dict_primal : dict 
        Dictionary of dictionaries. See the structure below
            dict_primal = {
                        'inner_dict_1': {
                            'df_trt_1': df1,
                            'df_trt_2': df2
                        },
                        'inner_dict_2': {
                            'df_trt_3': df3
                        }
                    }
        Each one of the inner dictionaries contains a set of Data Frames, each Data Frame corresponds to a single treatment 

    Returns
    -------
    combined_dict : dict
        Combined dictionary from dict_primal. See the structure below
            combined_dict = {
                            'df_trt_1': df1,
                            'df_trt_2': df2
                            'df_trt_3': df3
                        }
    '''
    combined_dict = {}
    for keys in list(dict_primal.keys()):
        for name, df in dict_primal[keys].items():
            combined_dict[name] = df
    return combined_dict




def remove_undesired_data_custom(outer_dict, shift):
    for key in outer_dict:
        inner_dict = outer_dict[key]
        outer_dict[key] = {df_name: df.iloc[:df[df['Clotting_2'] == 1].index[0] + shift]
                           for df_name, df in inner_dict.items()
                          }

