#!/usr/bin/env python
import argparse
from plnn.model import load_and_simplify

import gurobipy as grb
import torch

from torch import nn
from plnn.network_linear_approximation import LinearizedNetwork


class VerificationHelper(object):
    def __init__(self, gmodel):
        self.gmodel = gmodel

    def _dense_vars(self, layer):
        output_size = layer.out_features
        epsilon_vars = []
        output_vars = []
        delta_vars = []
        for i in range(output_size):
            epsilon_vars.append(self.gmodel.addVar())
            output_vars.append(self.gmodel.addVar(lb=-grb.GRB.INFINITY))
            delta_vars.append(self.gmodel.addVar(vtype=grb.GRB.BINARY))
        return epsilon_vars, output_vars, delta_vars

    def _dense_constraints(self, layer, epsilons, inputs, outputs):
        output_size = layer.out_features
        weights = layer.weight.data.numpy()
        bias = layer.bias.data.numpy()
        dotted_outputs = []

        for i in range(output_size):
            next_output = weights[i].dot(inputs) + bias[i]
            dotted_outputs.append(next_output)

        for i in range(output_size):
            self.gmodel.addConstr(dotted_outputs[i] - outputs[i] <= epsilons[i])
            self.gmodel.addConstr(dotted_outputs[i] - outputs[i] >= -epsilons[i])

    def _relu_vars(self, layers, idx):
        output_size = layers[idx - 1].out_features
        relu_vars = []
        for i in range(output_size):
            relu_vars.append(self.gmodel.addVar())
        return relu_vars

    def _relu_constraints(self, layers, idx, pre, post, delta):
        output_size = layers[idx - 1].out_features
        for i in range(output_size):
            # self.gmodel.addGenConstrMax(post[i], [pre[i]], 0)
            self.gmodel.addConstr(post[i] >= pre[i])
            self.gmodel.addConstr(post[i] <= pre[i] + 1000 * delta[i])
            self.gmodel.addConstr(post[i] <= 1000 * (1 - delta[i]))

    def add_vars(self, layers):
        dense, relu = [], []
        for layer_idx, layer in enumerate(layers):
            if type(layer) is nn.Linear:
                dense.append(self._dense_vars(layer))
            elif type(layer) is nn.ReLU:
                relu.append(self._relu_vars(layers, layer_idx))
        # for i in range(0, len(layers) - 1, 2):
        #     dense.append(self._dense_vars(layers[i]))
        #     relu.append(self._relu_vars(layers[i + 1]))
        # dense.append(self._dense_vars(layers[-1]))
        return dense, relu

    def add_constraints(self, layers, il, dense, relu):
        for i in range(0, len(relu)):
            e, o, d = dense[i]
            r = relu[i]

            self._dense_constraints(layers[2 * i], e, il, o)
            self._relu_constraints(layers, 2 * i + 1, o, r, d)
            il = r
        (e, o, _) = dense[-1]
        self._dense_constraints(layers[-1], e, il, o)
        return o


class MIPLalNetwork:
    def __init__(self, layers):
        '''
        layers: A list of Pytorch layers containing only Linear/ReLU/MaxPools
        '''
        self.layers = layers
        self.net = nn.Sequential(*layers)

        # Initialize a LinearizedNetwork object to determine the lower and
        # upper bounds at each layer.
        self.lin_net = LinearizedNetwork(layers)

    def solve(self, inp_domain):
        '''
        inp_domain: Tensor containing in each row the lower and upper bound
                    for the corresponding dimension

        Returns:
        sat     : boolean indicating whether the MIP is satisfiable.
        solution: Feasible point if the MIP is satisfiable,
                  None otherwise.
        '''

        self.gurobi_vars = []
        self.model = grb.Model()
        helper = VerificationHelper(self.model)

        # First add the input variables as Gurobi variables.
        inp_gurobi_vars = []
        for (lb, ub) in inp_domain:
            v = self.model.addVar(lb=lb, ub=ub)
            inp_gurobi_vars.append(v)

        self.gurobi_vars.append(inp_gurobi_vars)

        # Add all the variables to the network.
        dense, relu = helper.add_vars(self.layers)

        # Set the objective to be the sum of the epsilons.
        objective = grb.quicksum((grb.quicksum(e) for (e, _, _) in dense))
        self.model.setObjective(objective, grb.GRB.MINIMIZE)
        self.model.update()

        # Add the constraints for the network itself.
        self.gurobi_vars.append(helper.add_constraints(self.layers, inp_gurobi_vars, dense, relu))

        # Assert that this is as expected: a network with a single output
        assert len(self.gurobi_vars[-1]) == 1, "Network doesn't have scalar output"

        # TODO: What determines this function?
        # output_constraint_fn = lambda l, r: r <= l

        # Then add the output constraints and any extra input constraints.
        # TODO: Choose appropriate function
        self.model.addConstr(self.gurobi_vars[-1][-1] <= 0)

        # Specify the cutoff of 1e-6.
        self.model.Params.Cutoff = 1e-6

        # Perform the optimisation.
        self.model.optimize()

        # Print the result.
        if self.model.status == grb.GRB.OPTIMAL:
            # print("Environment state {} giving outputs ({}, {})".format([e.x for e in inp_domain], left.x, right.x))
            len_inp = len(self.gurobi_vars[0])

            # Get the input that gives the feasible solution.
            inp = torch.Tensor(len_inp)
            for idx, var in enumerate(self.gurobi_vars[0]):
                inp[idx] = var.x
            print("Counter-example found.")
            print(inp)
            return (True, inp)
        elif self.model.status == grb.GRB.CUTOFF or self.model.status == grb.GRB.INFEASIBLE:
            print("No counter-example could be found.")
            return (False, None)


def main():
    parser = argparse.ArgumentParser(description="Read a .rlv file"
                                     "and prove its property.")

    parser.add_argument('rlv_infile', type=argparse.FileType('r'),
                        help='.rlv file to prove.')
    args = parser.parse_args()

    mip_network, domain = load_and_simplify(args.rlv_infile,
                                            MIPLalNetwork)
    sat, solution = mip_network.solve(domain)

    if sat is False:
        print("UNSAT")
    else:
        print("SAT")
        #print(solution)


if __name__ == '__main__':
    main()
