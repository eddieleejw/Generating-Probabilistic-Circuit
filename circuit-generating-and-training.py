import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import math
import torch.nn as nn
import time
import random
import numpy as np


def read_data_into_tensor(input_text_file, batchsize):
    data = open(str(input_text_file), "r")
    data_lines_text = data.readlines() #create list, where each element is a string of data
    data_lines_list = []
    for x in range(0, len(data_lines_text)):
        data_lines_list.append([])
        for y in range(0, len(data_lines_text[x])):
            if (y % 2 == 0):
                data_lines_list[x].append(float(data_lines_text[x][y]))
    batched_data_lines_list = []
    n = 0
    i = -1

    for data in data_lines_list:
        if(n%batchsize == 0):
            batched_data_lines_list.append([])
            i += 1
        batched_data_lines_list[i].append(data)
        n += 1

    data_lines_tensor = torch.tensor(batched_data_lines_list)
    return data_lines_tensor


def loss_function(y_pred):
    return -torch.mean(y_pred)


class MyDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.int)
        self.x = torch.from_numpy(xy)
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.n_samples


class product_node:
    def __init__(self):
        self.children = []
        self.type = "P"


class sum_node:
    def __init__(self):
        self.children = []
        self.type = "S"


class input_node:
    def __init__(self):
        self.name = ""
        self.type = "I"


def split_list(input_list, number_of_chunks):  # for splitting up a list into roughly evenly sized chunks
    '''
    This function takes an input_list, and creates n roughly equal sized lists which sample from input_list without replacement
    :param input_list: the list to sample/split
    :param number_of_chunks: number of lists we want at the end
    :return: list containing each sublist
    '''
    length_of_list = len(input_list)
    quotient = length_of_list // number_of_chunks
    quotient_plus_one = quotient + 1

    split_index = []  # this list will hold integers, which represent how the input_list will be divided e.g. a list of size 10 into 3 chunks could have split index [4,3,3]
    items_left = length_of_list
    spaces_left = number_of_chunks
    while ((items_left - quotient_plus_one) >= (
            spaces_left - 1) * quotient):  # if this condition is true,it means that we can still afford to add another chunk of size quotient_plus_one
        split_index.append(quotient_plus_one)
        items_left -= quotient_plus_one
        spaces_left -= 1
    while (number_of_chunks != len(split_index)):  # fill the rest of the list with quotient
        split_index.append(quotient)

    shuffled_list = input_list.copy()
    random.shuffle(shuffled_list)

    list_of_list = []
    for n in split_index:  # according to split_index, compile list_of_list
        temp_list = []
        for m in range(n):
            temp_list.append(shuffled_list.pop())
        list_of_list.append(temp_list)

    return list_of_list


def read_tree_struc(input_text, tree_dict):
    input_tree_text = open(str(input_text), "r")
    for node in input_tree_text.readlines():
        details = node.split()
        if (details[0] == "P"):
            tree_dict[int(details[1])] = product_node()
            temp_count = 1
            for i in details:
                if (temp_count > 2):
                    (tree_dict[int(details[1])].children).append(int(i))
                temp_count += 1
        elif (details[0] == "S"):
            tree_dict[int(details[1])] = sum_node()
            temp_count = 1
            for i in details:
                if (temp_count > 2):
                    (tree_dict[int(details[1])].children).append(int(i))
                temp_count += 1
        elif (details[0] == "I"):
            tree_dict[int(details[1])] = input_node()
            (tree_dict[int(details[1])]).name = details[2]


def write_tree_struc(tree_dict, file_name):
    output_file = open(file_name, "w")
    for ID in tree_dict:
        if ((tree_dict[ID]).type == "P" or (tree_dict[ID]).type == "S"):
            output_file.write(str((tree_dict[ID]).type) + " " + str(ID))
            for child in (tree_dict[ID]).children:
                output_file.write(" " + str(child))
        elif ((tree_dict[ID]).type == "I"):
            output_file.write("I " + str(ID) + " " + (tree_dict[ID]).name)
        output_file.write("\n")


def create_product_node_2(tree_dict, scope):

    max_child_nodes = 4
    max_child_nodes = min(len(scope), max_child_nodes)
    number_of_child_nodes = random.randrange(2, max_child_nodes + 1)

    # number_of_child_nodes = len(scope)

    split_scope = split_list(scope, number_of_child_nodes)  # split the list

    children_list = []
    for num in range(number_of_child_nodes):
        temp_ID = create_sum_node(tree_dict, split_scope[num])
        children_list.append(temp_ID)

    temp_node = product_node()
    temp_node.children = children_list

    new_ID = list(tree_dict)[-1] + 1
    tree_dict[new_ID] = temp_node
    return new_ID


def create_sum_node(tree_dict, scope):
    if len(scope) == 1:
        temp_node = sum_node()
        children_list = [int(2 * scope[0]), int(2 * scope[0] + 1)]
        temp_node.children = children_list

    else:
        number_of_child_nodes = 2

        children_list = []
        for num in range(0, number_of_child_nodes):
            temp_ID = create_product_node_2(tree_dict, scope)
            children_list.append(temp_ID)

        temp_node = sum_node()
        temp_node.children = children_list

    new_ID = list(tree_dict)[-1] + 1
    tree_dict[new_ID] = temp_node
    return new_ID


def generate_circuit_new(tree_dict, input_node_num):
    scope_list = []
    for x in range(0, input_node_num):
        for y in range(0, 2):
            temp_input_node = input_node()
            if (y == 0):
                temp_input_node.name = "X_" + str(x + 1)
            else:
                temp_input_node.name = "B_X_" + str(x + 1)  # "B_" indicating 'bar'.
            temp_num = int(
                2 * x + y)  # assigns each input node a unique identifier starting from 0, up to 2n - 1, where n is the number of input nodes
            tree_dict[temp_num] = temp_input_node
        scope_list.append(x)
    # now scope_list = [0,1,..., n-1] and tree_dict = {0:X_1, 1: B_X_1, ...}
    create_sum_node(tree_dict, scope_list)


class PCCircuit(nn.Module):  # inherit from torch.nn.Module
    def __init__(self, tree_dict, circuit_type, parameter_value_name):
        '''
        :param: circuit_type: "test" or "train", depending on what kind of circuit this is
        :param: parameter_value_name: name of the text file which holds the parameter values
        '''
        super().__init__()  # use constructor of nn.Module

        self.tree_dict = tree_dict  # tree_dict represents circuit structure
        self.top_node = list((self.tree_dict).keys())[-1]

        self.weight_dict = {}
        self.create_weight_dict(self.top_node, tree_dict, self.weight_dict)

        self.number_of_weights = len(self.weight_dict)

        if circuit_type == "train":
            W = torch.randn(self.number_of_weights)
            self.W = nn.Parameter(W, requires_grad=True)
        elif circuit_type == "test" or circuit_type == "more_train":
            W = torch.tensor(self.read_parameter_values(str(parameter_value_name)), requires_grad=True)
            self.W = nn.Parameter(W, requires_grad=True)
        else:
            print("ERROR: circiut is neither training nor testing")

    def read_parameter_values(self, file_name):
        input_file = open(file_name, "r")
        L = []  # list
        S = ""  # string
        for char in input_file.read():
            if char not in [' ', "\n"]:
                if char == ',':  # reset string
                    L.append(float(S))
                    S = ""
                else:
                    S = S + char
        L.append(float(S))
        return L

    def create_weight_dict(self, node_ID, tree_dict, weight_dict):
        '''
        creates the weight_dict based on tree_dict. recursive function that travels to every node, and if it is a sum node, it adds the corresponding
        key and value to dictionary

        :param node_ID: ID of current node
        :param tree_dict: the circuit structure
        :param weight_dict: the current weight_dict
        :return: a dictionary
        '''

        if (tree_dict[node_ID].type == "S"):  # this is a sum node. update weight_dict
            for n in range(len(
                    (self.tree_dict)[node_ID].children)):  # loop from n = 0, 1, ... for every child of this sum node
                try:
                    weight_dict[(node_ID, n)] = list(weight_dict.values())[
                                                    -1] + 1  # set the value to be whatever number is not in the values yet
                except IndexError:
                    weight_dict[(node_ID, n)] = 0

            for child in (self.tree_dict)[node_ID].children:  # recursive call on each child
                self.create_weight_dict(child, tree_dict, weight_dict)

        elif (tree_dict[node_ID].type == "P"):
            for child in (self.tree_dict)[node_ID].children:  # recursive call on each child
                self.create_weight_dict(child, tree_dict, weight_dict)

    def recursive_evaluation(self, tree_dict, node_ID,
                             x):  # initially, layernum should be height-1, and node_num should be 0
        '''

        :param tree_dict: dictionary which holds tree structure
        :param node_ID: current node that is being evaluated
        :param input_evaluations: a 2-d tensor which holds some 'batchsize' number of data points
        :return: a 2-d tensor which holds a 'batchsize' number of scalars
        '''
        if (tree_dict[node_ID].type == "I"):  # we have arrived at input node

            # e.g. 0/1 = (B)X_1: 0. 2/3 = (B)X_2: 1, etc. corresponds to which index in each data point is correct for this node's scope
            index = node_ID // 2

            # create index_tensor, a 2-d tensor that holds the correct index of the data "x" that we want
            index_tensor = torch.tensor([])

            for y in x:
                index_tensor = torch.cat((index_tensor, torch.tensor([index])))

            index_tensor = index_tensor.view(-1, 1)
            index_tensor = index_tensor.type(torch.int64)

            x_gathered = torch.gather(x, 1, index_tensor)
            x_gathered = torch.squeeze(x_gathered)

            if (
                    node_ID % 2 == 0):  # if nodenum is even, meaning that this node should evaluate to directly corresponding data in x
                return torch.log(x_gathered)
            else:  # nodenum is odd, so this node outputs (1 - corresponding data in x)
                return torch.log(1 - x_gathered)

        elif (tree_dict[node_ID].type == "P"):  # product node

            temp_child_list = (self.tree_dict)[
                node_ID].children  # this will some list [0,1,5,6], which holds the node_ID of each child

            all_input_from_below = torch.tensor([])

            for child in temp_child_list:  # iterating through the node_ID of each child

                input_from_below = self.recursive_evaluation(tree_dict, child,
                                                             x)  # this is the tensor we recieve from this recursive call. this is already in log
                input_from_below = torch.unsqueeze(input_from_below, 0)

                all_input_from_below = torch.cat((all_input_from_below, input_from_below),
                                                 dim=0)  # concatenate like so [[x_1], [x_2], ..., [x_n]]

            final = torch.sum(all_input_from_below, dim=0)

            return final



        elif (tree_dict[node_ID].type == "S"):

            temp_child_list = (self.tree_dict)[
                node_ID].children  # this will some list [0,1,5,6], which holds the node_ID of each child

            weights_IDs = [self.weight_dict[(node_ID, n)] for n in range(len(temp_child_list))]
            weights_IDs = torch.tensor(weights_IDs)
            weights_list = torch.gather(self.W, 0, weights_IDs)

            self.W.requires_grad
            weights_list.requires_grad


            softmax = nn.Softmax(dim=0)
            weights_list = softmax(weights_list)


            log_weights_list = torch.log(weights_list)

            all_input_from_below = torch.tensor([])

            for child in temp_child_list:  # iterating through the node_IDs

                input_from_below = self.recursive_evaluation(tree_dict, child,
                                                             x)  # this is the tensor we recieve from this recursive call. this is already in log
                input_from_below = torch.unsqueeze(input_from_below, 0)

                all_input_from_below = torch.cat((all_input_from_below, input_from_below),
                                                 dim=0)  # concatenate like so [[x_1], [x_2], ..., [x_n]]

            all_input_from_below = torch.transpose(all_input_from_below, 0, 1)
            log_x_plus_log_weights = torch.add(all_input_from_below,
                                               log_weights_list)  # for each data in all_input_from_below, add the respective log weights
            # the above is a tensor that loosk like [ [...], [...], ... ], where all the inner tensors already have the log(weights) applied inside

            final = torch.logsumexp(log_x_plus_log_weights,
                                    1)  # this computes logsumexp for each data within the tensor

            return final



        else:
            print("Error (evaluate_node)")

    def forward(self, x):

        result = self.recursive_evaluation(self.tree_dict, self.top_node, x)

        return result


'''
Tweak some parameters here
'''
# type is either 0:"train" or 1:"test" or 2: continue training
circuit_type = 0

file_name = "data.txt"
parameter_file_name = "param_value.txt"
tree_struc_file_name = "tree_structure.txt"

if circuit_type == 0:
    number_of_epochs = 20
    learning_rate = 0.01
    generate_new_circuit = True
elif circuit_type == 1:
    generate_new_circuit = False
elif circuit_type == 2:
    generate_new_circuit = False
    learning_rate = 0.01
    number_of_epochs = 5

'''
Don't change these
'''
if circuit_type == 0:
    circuit_type = "train"
elif circuit_type == 1:
    circuit_type = "test"
elif circuit_type == 2:
    circuit_type = "more_train"
else:
    print("ERROR: circuit_type is not 0 or 1")
    exit()

counter = 0
torch.set_printoptions(threshold=10000000000000)

if circuit_type == "train" or circuit_type == "more_train":
    dataset = MyDataset(file_name)
    dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True, num_workers=2)
    dataiter = iter(dataloader)
    print_inner_counter = 5  # print loss.item() every n batches

elif circuit_type == "test":
    testfile = open(file_name, "r")
    batchsize = len(testfile.readlines())
    datatensor = read_data_into_tensor(file_name, batchsize)
    number_of_epochs = 1  # determine how many epochs
    print_inner_counter = 2  # print loss.item() every n batches
    learning_rate = 0
else:
    print("ERROR: type is neither \"train\" nor \"test\"")

if generate_new_circuit:
    openfile = open(file_name, "r")
    length = len(openfile.readline())
    number_of_input_variables = int(length / 2)
    print(f"no. of variables = {number_of_input_variables}")

    tree_dict = {}
    generate_circuit_new(tree_dict, number_of_input_variables)

    number_of_nodes = list(tree_dict)[-1]
    # if number_of_nodes < number_of_input_variables*30:

    write_tree_struc(tree_dict, tree_struc_file_name)
    print(f"There are {number_of_nodes} nodes")

'''
Read structure
'''
tree_dict = {}
read_tree_struc(tree_struc_file_name, tree_dict)

'''
Create model
'''
model = PCCircuit(tree_dict, circuit_type, parameter_file_name)  # create our model
print(f"Number of weights in this model is {model.number_of_weights}")
print(f"DEBUG: learning_rate = {learning_rate}")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if circuit_type == "train" or circuit_type == "more_train":
    while (counter < number_of_epochs):  #
        counter += 1
        average = []

        inner_counter = 0

        # for data in datatensor:
        for i, data in enumerate(dataloader):

            start = time.process_time()

            y_pred = model(data)  # forward pass

            loss = loss_function(y_pred)
            optimizer.zero_grad()  # zero the gradient
            loss.backward()
            optimizer.step()
            average.append(loss.item())
            if inner_counter % print_inner_counter == 0:
                print(loss.item())

            print(f"Time taken per batch = {time.process_time() - start}.")
            inner_counter += 1

        print(f"Average log likelihood of iteration number {counter} is: {sum(average) / len(average)}")

        with open(parameter_file_name, "w") as f:
            f.write((str(model.W.data))[8:-2])

elif circuit_type == "test":
    for data in datatensor:
        y_pred = model(data)  # forward pass
        loss = loss_function(y_pred)
        print(f"Average log likelihood is {loss.item()}")



