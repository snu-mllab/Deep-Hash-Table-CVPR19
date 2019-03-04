import numpy as np

def content_matrix_write(matrix, f):
    nrow = len(matrix)
    ncol = len(matrix[0])
    max_col_size = list()
    for c in range(ncol):
        tmp = 0
        for r in range(nrow):
            col_size = len(matrix[r][c])
            if tmp < col_size: tmp=col_size
        max_col_size.append(tmp)

    for c in range(ncol):
        for r in range(nrow):
            matrix[r][c]=" "*(max_col_size[c]-len(matrix[r][c]))+matrix[r][c]

    for r in range(nrow):
        f.write(','.join(matrix[r])+'\n')

def write_multi_array(dict_, path):
    keys = [str(v) for v in dict_.keys()]
    content_matrix = list()

    content_matrix.append(keys)
    length = len(dict_[keys[0]])
    for idx in range(length):
        tmp_row = list()
        for key in keys:
            tmp_row.append(str(dict_[key][idx]))
        content_matrix.append(tmp_row)

    with open(path, 'w') as f:
        content_matrix_write(content_matrix, f)

def write_dict_csv(dict_, path):
    print("Write csv files on %s"%path)
    with open(path, 'w') as f:
        for key in sorted(dict_.keys()): f.write('{},{}\n'.format(str(key), dict_[key]))

class CsvWriter:
    def __init__(self, nline):
        self.nline = nline
        self.header_set = list()
        self.content_set = list()
        for i in range(self.nline):
            self.content_set.append(list())

    def add_header(self, header):
        self.header_set.append(header)
    
    def add_content(self, index, content):
        self.content_set[index].append(content)

    def write(self, path):
        print("Write csv files on %s"%path)
        with open(path, 'w') as f:
            f.write(','.join([str(v) for v in self.header_set])+'\n')
            for i in range(self.nline):
                f.write(','.join([str(v) for v in self.content_set[i]])+'\n')

class CsvWriter2:
    def __init__(self, nline):
        self.nline = nline
        self.header_set = list()
        self.content_set = list()
        for i in range(self.nline):
            self.header_set.append(list())
            self.content_set.append(list())

    def add_header(self, index, header):
        self.header_set[index].append(header)
    
    def add_content(self, index, content):
        self.content_set[index].append(content)

    def write(self, path):
        print("Write csv files on %s"%path)
        
        max_length = 0
        for i in range(self.nline):
            max_length = max(max_length, max([len(str(v)) for v in self.header_set[i]]))
            max_length = max(max_length, max([len(str(v)) for v in self.content_set[i]]))

        with open(path, 'w') as f:
            for i in range(self.nline):
                f.write(','.join([str(v).rjust(max_length, ' ') for v in self.header_set[i]])+'\n')
                f.write(','.join([str(v).rjust(max_length, ' ') for v in self.content_set[i]])+'\n')

class CsvWriter3:
    def __init__(self):
        self.content_dict = dict()

    def add_content(self, row_key, column_key, content):
        if row_key not in self.content_dict.keys():
            self.content_dict[row_key] = dict()
        self.content_dict[row_key][column_key] = content

    def write(self, path, row_key_set, col_key_set):
        print("Write csv files on %s"%path)
        
        nrow = len(row_key_set)

        col_header = list()
        col_header.append('')
        for col_key in col_key_set:
            col_header.append(str(col_key))

        content_list = list()
        for row_key in row_key_set:
            tmp = list()
            tmp.append(str(row_key))
            for col_key in col_key_set:
                if col_key in self.content_dict[row_key].keys():
                    tmp.append(self.content_dict[row_key][col_key])
                else:
                    tmp.append('-')
            content_list.append(tmp) 

        max_len_list = list()
        nrow, ncol = len(content_list), len(content_list[0])

        for c_idx in range(ncol):
            tmp = len(col_header[c_idx])
            for r_idx in range(nrow):
                tmp = max(tmp, len(str(content_list[r_idx][c_idx])))
            max_len_list.append(tmp) # max_len_list[c_idx] = tmp

        with open(path, 'w') as f:
            f.write(','.join([str(col_header[c_idx]).rjust(max_len_list[c_idx], ' ') for c_idx in range(ncol)])+'\n')
            for r_idx in range(nrow):
                f.write(','.join([str(content_list[r_idx][c_idx]).rjust(max_len_list[c_idx], ' ') for c_idx in range(ncol)])+'\n')

def read_csv(path):
    '''
    Complementary to CsvWriter
    '''
    content_set = list()
    with open(path, 'r') as lines:
        reads = list()
        for line in lines: 
            reads.append(line)
        header_set = reads[0][:-1]
        for read in reads[1:]:
            content_set.append(read[:-1].split(','))
    return header_set, content_set
