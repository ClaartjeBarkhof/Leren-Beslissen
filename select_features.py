import numpy as np
import main

trails = 1

def powerset(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]

def compute_error(option):
    all_errors = []
    for i in range(trails):
        print(option)
        error = main.compute_error(option)
        all_errors.append(error)
    return all_errors, np.mean(all_errors)

def pick_best_opt(option_list):
    lowest_error = 999
    best_option = None
    for option in option_list:
        all_errors, error = compute_error(option)
        fh = open("options.txt", "a") 
        fh.write('"'+str(option)+'"' + "," +'"'+str(error)+'"'+','+'"'+str(all_errors)+'"'+"\n") 
        fh.close 
        if error < lowest_error:
            lowest_error = error
            best_option = option
    return best_option

def pick_best_feature(option_list):
    f = open("subsets.txt", "w") 
    f.write("") 
    f.close 
    lowest_error = 999
    best_option = None
    for option in option_list:
        if len(option) == 1:
            continue
        all_errors, error = compute_error(option) 
        fh = open("subsets.txt", "a") 
        fh.write('"'+str(option)+'"' + "," +'"'+str(error)+'"'+','+'"'+str(all_errors)+'"'+"\n") 
        fh.close 
        if error < lowest_error:
            lowest_error = error
            best_option = option
    return best_option
    
def pick_options():
    options_all = [[['descr_tfidf', 'shipping', 'item_condition', 'brand_nothing', 'cat_1'],
    ['descr_sentiment', 'shipping', 'item_condition', 'brand_nothing', 'cat_1'],
    ['descr_len', 'shipping', 'item_condition', 'brand_nothing', 'cat_1']],
    [['name_bin', 'shipping', 'item_condition', 'brand_nothing', 'cat_1'],
    ['name_tfidf', 'shipping', 'item_condition', 'brand_nothing', 'cat_1']],
    [['brand_fill', 'shipping', 'item_condition', 'cat_1'],
    ['brand_nothing', 'shipping', 'item_condition', 'cat_1']],
    [['cat_1', 'shipping', 'item_condition', 'brand_nothing'],
    ['cat_2', 'shipping', 'item_condition', 'brand_nothing'],
    ['cat_3', 'shipping', 'item_condition', 'brand_nothing'],
    ['cat_4', 'shipping', 'item_condition', 'brand_nothing'],
    ['cat_all', 'shipping', 'item_condition', 'brand_nothing']]]    
    best_opt = []
    f = open("options.txt", "w") 
    f.write("") 
    f.close 
    for option_list in options_all:
        best_option = pick_best_opt(option_list)
        best_opt.append(best_option[0])
    print(best_opt)
    return best_opt

def find_features():
    best_opt = pick_options()
    print("selected:", best_opt)
    main_features = ["shipping", "item_condition_id", best_opt[0], best_opt[1], best_opt[2],best_opt[3]]
    subsets = list(powerset(main_features))[1:]
    best_feature_set = pick_best_feature(subsets)
    print("best features", best_feature_set)
    return best_feature_set

find_features()