import numpy as np
import copy
from sklearn import linear_model
from docplex.mp.model import Model
from Models.Cursor import Cursor
from scipy.special import comb
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

"""
     Wrapper class for optimizers methods passed into the CSMIO object.
"""


def get_equation(betas, terms, cur_dimension):
    output_equation = "f_%s(x) = " % (cur_dimension + 1)
    bias = 0
    output = []
    for i in range(len(terms)):
        if terms[i] == "Bias":
            bias = betas[i]
        else:
            if betas[i] > 0:
                output.append("+%.8f%s" % (betas[i], terms[i]))
            elif betas[i] < 0:
                output.append("%.8f%s" % (betas[i], terms[i]))

    output_equation += "".join(output)
    if bias > 0:
        output_equation += "+%.8f" % bias
    elif bias < 0:
        output_equation += "%.8f" % bias

    return output_equation


def generate_train_data(raw_feature, raw_response, term_index, cur_dimension):
    output_feature = np.ones((raw_feature.shape[0], len(term_index)), dtype=float)
    col = 0
    d = raw_feature.transpose()
    while col < len(term_index):
        for item in term_index[col]:
            output_feature[:, col] = output_feature[:, col] * d[item, :]
        col += 1
    output_response = raw_response[:, cur_dimension]

    return output_feature, output_response


def generate_all_train_data(raw_feature, terms):
    output_feature = np.ones((raw_feature.shape[0], len(terms)), dtype=float)
    col = 0
    d = raw_feature.transpose()
    while col < len(terms):
        for item in terms[col]:
            output_feature[:, col] = output_feature[:, col] * d[item, :]
        col += 1

    return output_feature


def generate_all_train_data(raw_feature, terms):
    output_feature = np.ones((raw_feature.shape[0], len(terms)), dtype=float)
    col = 0
    d = raw_feature.transpose()
    while col < len(terms):
        for item in terms[col]:
            output_feature[:, col] = output_feature[:, col] * d[item, :]
        col += 1

    return output_feature


def print_term(term):
    output = ''
    term_set = set(term)
    for x in term_set:
        power = term.count(x)
        if power == 1:
            output += ('X%d' % x)
        else:
            output += ('X%d^%d' % (x, power))
    return output


def get_terms(term_list, index):
    if index < 0:
        return 'Bias'
    else:
        output = print_term(term_list[index])
        return output


def square_of_list(input_list):
    return sum([i ** 2 for i in input_list])


class CSMIOOptimizer:

    def __init__(
            self,
            dimension=3,
            order=2,
            num_candidate_terms=100,
            lasso_alpha=0.000001,
            intercept=True,
            term_ks=[],
            betas_ub=1000,
            betas_lb=-1000,
            time_limit=600,
            mip_gap=0.0,
            mip_detail=False,
    ):

        log_dir = './Log/'
        log_file = 'log.txt'
        self.f_log = open(log_dir + log_file, 'w')
        if dimension < 0:
            raise ValueError("dimension cannot be negative")

        if order < 0:
            raise ValueError("order cannot be negative")

        if num_candidate_terms < 0:
            raise ValueError("number of candidate terms cannot be negative")

        if time_limit < 0:
            raise ValueError("time limit of MIP solver cannot be negative")

        if betas_lb >= betas_ub:
            raise ValueError("upper bound of betas should be larger than lower bound of betas")

        self.order = order
        self.dimension = dimension
        self.num_candidate_terms = num_candidate_terms
        self.lasso_alpha = lasso_alpha
        self.term_ks = term_ks
        self.intercept = intercept
        self.betas_ub = betas_ub
        self.betas_lb = betas_lb
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.mip_detail = mip_detail

        self.f_log.write('************** Parameters  ********************************************\n')
        for attr, value in self.__dict__.items():
            if attr == 'f_log':
                continue
            self.f_log.write('%s : %s\n' % (attr, value))
        self.f_log.write('\n \n')

    coefficients = []
    features = []

    def fit(self, x, y):
        feature = x
        response = y

        # store all the nonzero coefficients of equations
        all_equations = []  # fitted equations

        print('start fit')
        pool_terms_backup = []
        for cur_order in range(1, self.order + 1):
            # Calculate total count of terms in current order
            term_total_counts = int(comb(self.dimension, cur_order, False, True))
            cursor = Cursor(cur_order)
            selected_term_index = cursor.forward(term_total_counts, self.dimension)
            pool_terms_backup.extend(selected_term_index)
            cur_order += 1

        pool_terms = copy.deepcopy(pool_terms_backup)

        # generate data for terms selection
        pool_term_feature = generate_all_train_data(feature, pool_terms)

        for cur_dimension in range(self.dimension):
            print('************** Equation %d ******************************************' % (cur_dimension + 1))
            pool_terms = copy.deepcopy(pool_terms_backup)

            # Step 1: ****** candidate terms selection ************** #
            if len(pool_terms) > self.num_candidate_terms:

                pool_term_response = response[:, cur_dimension]
                lasso_model = make_pipeline(StandardScaler(), linear_model.LassoLars(alpha=self.lasso_alpha,normalize=False))
                #lasso_model = linear_model.LassoLars(alpha=self.lasso_alpha)
                lasso_model.fit(pool_term_feature, pool_term_response)


                # sort by absolute value in descending direction
                arg_sorted = np.argsort(-np.abs(lasso_model[-1].coef_))
                pool_terms.clear()

                for i in arg_sorted[0: self.num_candidate_terms]:
                    if lasso_model[-1].coef_[i] != 0:
                        pool_terms.append(pool_terms_backup[i])

            self.features.append(pool_terms)
            # Step 2: ****** discrete optimization ************** #
            new_pool_term_feature, new_pool_term_response = generate_train_data(feature, response, pool_terms,
                                                                                cur_dimension)

            if len(self.term_ks) < self.dimension:
                term_k = self.term_ks[0]
            else:
                term_k = self.term_ks[cur_dimension]

            output = self.discrete_optimization(new_pool_term_feature, new_pool_term_response, pool_terms,
                                                term_k, cur_dimension)

            # store all the fitted equations
            print(output[0])
            all_equations.append(output[0])
            self.coefficients.append(output[2])

        self.f_log.write('************** Outputs  ********************************************\n')
        for eq in all_equations:
            self.f_log.write(eq)
            self.f_log.write('\n')

    def discrete_optimization(self, all_features, all_responses, term_list, true_term, cur_dimension):
        all_features = np.c_[np.ones(all_features.shape[0]), all_features]
        features = all_features
        responses = all_responses

        # count of coefficients including Beta0
        count = features.shape[1]
        n_data = features.shape[0]

        # construct model
        model = Model(name='MIQP_PDERegression', cts_by_name=True, log_output=self.mip_detail)
        model.parameters.mip.tolerances.mipgap = self.mip_gap
        model.parameters.timelimit = self.time_limit

        betas_index = [i for i in range(0, count)]
        alphas_index = [i for i in range(0, count)]
        betas = model.continuous_var_list(betas_index, name='beta', ub=self.betas_ub, lb=self.betas_lb)
        alphas = model.binary_var_list(alphas_index, name='alpha')

        # Constraint: |Beta_i| <= M Alpha_i by using indicator
        for i in range(count):
            model.add_indicator(alphas[i], betas[i] == 0.0, 0)

        # Constraint: Sum(Alpha_i) == k
        model.add_constraint(model.sum(alphas[j] for j in alphas_index) == true_term, ctname='equal_k')

        # beta constraints
        if not self.intercept:
            alphas[0].lb = 0
            alphas[0].ub = 0

        # objective
        # yi^2 constant
        obj_constant = model.linear_expr(constant=square_of_list(responses))

        # linear
        fea_res = [features[:, i].dot(responses) for i in range(0, features.shape[1])]
        fea_res = [i * -2 for i in fea_res]
        obj_linear = model.dot(betas, fea_res)

        # quadratic
        fea_square = [features[:, i].dot(features[:, i]) for i in range(0, features.shape[1])]
        betas_square = [betas[i] * betas[i] for i in range(count)]
        obj_quad_square = model.dot(betas_square, fea_square)

        betas_betas = [betas[i] * betas[j] for i in range(0, features.shape[1]) for j in
                       range(i + 1, features.shape[1])]
        fea_fea = [features[:, i].dot(features[:, j]) for i in range(0, features.shape[1]) for j in
                   range(i + 1, features.shape[1])]
        fea_fea = [i * 2 for i in fea_fea]
        obj_quad_beta = model.dot(betas_betas, fea_fea)

        # ridge weight
        lambda_ridge = np.std(responses) / np.sqrt(n_data)

        model.minimize(obj_constant + obj_linear + obj_quad_square + obj_quad_beta
                       + model.sum(betas_square) * lambda_ridge)

        # modify constraint of term number
        # print('Solving MIP model with k = %d' % true_term)
        sol = model.solve()
        # model.prettyprint()
        alphas_selected = []
        alphas_sol = sol.get_value_list(alphas)
        betas_sol = sol.get_value_list(betas)
        for index in range(0, count):
            if alphas_sol[index] == 1:
                alphas_selected.append(index)

        term_selected = [get_terms(term_list, i - 1) for i in alphas_selected]
        obj_value = sol.get_objective_value()

        # return the solution of least squares
        X_bar = features[:, alphas_selected]
        ls_model = linear_model.LinearRegression(fit_intercept=False)
        ls_model.fit(X_bar, responses)
        for i in range(len(alphas_selected)):
            betas_sol[alphas_selected[i]] = ls_model.coef_[i]

        betas_selected = [betas_sol[i] for i in alphas_selected]
        estimated_equation = get_equation(betas_selected, term_selected, cur_dimension)

        return estimated_equation, alphas_sol, [alphas_sol[i] * betas_sol[i] for i in range(len(betas_sol))]
