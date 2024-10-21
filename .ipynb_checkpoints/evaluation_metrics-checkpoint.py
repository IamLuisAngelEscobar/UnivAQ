import numpy as np
from tabulate import tabulate

def similarity(y_pred, y_true):
    if len(y_pred) != len(y_true):
        print('Here be dragons')
    else:
        comparison = np.equal(y_pred, y_true)
        score = np.sum(comparison)/len(y_pred)
    return score


def similarity_block(y_pred, y_true):
    if len(y_pred) != len(y_true):
        print('Here be dragons')
    else:
        #indices = y_pred == 1
        #comparison = y_true[indices]
        #score = np.sum(comparison)/len(indices)
        if np.sum(y_true) >= np.sum(y_pred):
            score = np.sum(y_pred)/np.sum(y_true)
        else:
            score = np.sum(y_true)/np.sum(y_pred)
    return score


def distance_pred(y_pred, y_true):
    first_nonzero_index_pred = np.nonzero(y_pred)[0][0]
    first_nonzero_index_true = np.nonzero(y_true)[0][0]
    diff = first_nonzero_index_pred-first_nonzero_index_true

    return diff


def performance(s, d, c):
    value = s*np.exp(-(d**2)/(2*c**2))
    if d >= 0:
        pass
    else:
        value = value * -1
    return value

def time_window_perform(v1, v2, window_size, individual):
    '''v1 must be y_pred'''
    half_window = window_size // 2
    comparisons = []
    TP, FP, TN, FN = 0, 0, 0, 0
    
    for i in range(len(v1)):
        # Determine the start and end of the window for v2
        start_idx = max(0, i - half_window)
        end_idx = min(len(v2), i + half_window)
        
        # Get the values from v2 for this window
        window_v2 = v2[start_idx:end_idx]
        comparisons.append((v1[i], window_v2))

        if v1[i] == 1:
            if 1 in (window_v2.values):
                TP += 1 #true positive
            else:
                FP += 1 #false positive
        else:
            if 1 in (window_v2.values):
                FN += 1 #false negative
            else:
                TN += 1 #true negative
    rows_matrix = [
        [0, TN, FP],
        [1, FN, TP]
    ]
    
    
    precision_block = TP/(TP+FP) if (TP+FP) != 0 else TP/(TP+FP+1)
    recall_block = TP/(TP+FN) if (TP+FN) != 0 else TP/(TP+FN+1)

    precision_noblock = TN/(TN+FN) if (TN+FN) != 0 else TN/(TN+FN+1)
    recall_noblock = TN/(TN+FP) if (TN+FP) != 0 else TN/(TN+FP+1)


    
    rows_report = [
        [0, precision_noblock, recall_noblock],
        [1, precision_block, recall_block]
    ]
    if individual == True:
        print('Confussion matrix:')
        print(tabulate(rows_matrix, headers=["True\\Prediction", "0", "1"], tablefmt="grid"))   
        print()
        print('Classification Report:')
        print(tabulate(rows_report, headers=[" ", "precision", "recall"], tablefmt="grid"))   
    

    return precision_noblock, recall_noblock, precision_block, recall_block

def oscillation_red(v1, threshold, window_size):
    v1_copy = v1.copy()
    for i in range(len(v1_copy)):
        #Determine the end of the window 
        end_idx = min(len(v1_copy), i + window_size)
        window_v1 = v1_copy[i:end_idx]
        criterion = oscillation_criteria(window_v1, threshold)
        if criterion == True:
            v1_copy[i:] = [1] * (len(v1_copy) - i)
            break
        else:
            pass
    return v1_copy


def oscillation_criteria(v1, threshold):
    total = np.sum(v1)/len(v1)
    ans = total >= (threshold)
    return ans


def class_report(p0, r0, p1, r1, stat):
    if stat == 'mean':
        rows_report = [
            [0, np.mean(p0), np.mean(r0)],
            [1, np.mean(p1), np.mean(r1)]
        ]
        
    elif stat == 'median':
        rows_report = [
            [0, np.median(p0), np.median(r0)],
            [1, np.median(p1), np.median(r1)]
        ]
        
    
    print('Classification Report:')
    print(tabulate(rows_report, headers=[" ", "precision", "recall"], tablefmt="grid"))  



