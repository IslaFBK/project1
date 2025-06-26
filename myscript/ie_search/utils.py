import brian2.numpy_ as np
'''
Judgement of parameter near or not
'''
def is_param_near(param, param_list, tol=0.05):
    # param: (ie_r_e1, ie_r_i1)
    for p in param_list:
        if np.linalg.norm(np.array(param) - np.array(p)) < tol:
            return True
    return False