import numpy as np
import main
from scipy import stats

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
        if len(option) < 3:
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
    #best_opt = pick_options()
    #print("selected:", best_opt)
    main_features = ["shipping", "item_condition", "descr_tfidf", "name_bin", "brand_fill", "cat_4"]
    subsets = list(powerset(main_features))[1:]
    best_feature_set = pick_best_feature(subsets)
    print("best features", best_feature_set)
    return best_feature_set

def wilcoxon(A, B):
    result = stats.wilcoxon(A, B)
    return result.pvalue

def show_significance():
    print("PICKING CATEGORIES")
    cat_3 =  ['cat_3', 'shipping', 'item_condition', 'brand_nothing']
    cat_4 = ['cat_4', 'shipping', 'item_condition', 'brand_nothing']
    cat_all = ['cat_all', 'shipping', 'item_condition', 'brand_nothing']
    cat_3_error, cat_3_error_list = main.compute_error(cat_3)[1:]
    print("Cat_3 error:", cat_3_error)
    cat_4_error, cat_4_error_list = main.compute_error(cat_4)[1:]
    print("Cat_4 error:", cat_4_error)
    cat_all_error, cat_all_error_list = main.compute_error(cat_all)[1:]
    print("Cat_all error:", cat_all_error)
    print("Comparing cat_3 and cat_4: p =", wilcoxon(cat_3_error_list, cat_4_error_list))
    print("Comparing cat_all and cat_4: p =", wilcoxon(cat_all_error_list, cat_4_error_list))
    print("\n")

    print("PICKING NAME")
    name_bin = ['name_bin', 'shipping', 'item_condition', 'brand_nothing', 'cat_1']
    name_tfidf = ['name_tfidf', 'shipping', 'item_condition', 'brand_nothing', 'cat_1']
    bin_error, bin_error_list = main.compute_error(name_bin)[1:]
    print("Binair error:", bin_error)
    nametf_error, nametf_error_list = main.compute_error(name_tfidf)[1:]
    print("Tfidf error:", nametf_error)
    print("Comparing bin and tfidf: p=", wilcoxon(bin_error_list, nametf_error_list))
    print("\n")

    print("PICKING FILLING IN BRAND NAMES OR NOT")
    fill = ['brand_fill', 'shipping', 'item_condition', 'cat_1'],
    no_fill = ['brand_nothing', 'shipping', 'item_condition', 'cat_1']
    fill_error, fill_error_list = main.compute_error(fill)[1:]
    no_fill_error, no_fill_error_list = main.compute_error(no_fill)[1:]
    print("Fill error:", fill_error)
    print("No fill error:", no_fill_error)
    print("Comparing fill and no fill: p=", wilcoxon(fill_error_list, no_fill_error_list))

show_significance()