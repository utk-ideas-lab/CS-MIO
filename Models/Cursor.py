import numpy as np


class Cursor:

    def __init__(self, order):
        self.cur_order = order
        self.cur_cursor = np.zeros(order, dtype=int)

    def forward(self, step, dimension):
        term_index = []
        for i in range(0, step):
            if self.cur_order == 1:
                term_index.append([self.cur_cursor[0]])
                self.cur_cursor[0] = (self.cur_cursor[0] + 1) % dimension

            elif self.cur_order == 2:
                term_index.append([self.cur_cursor[0], self.cur_cursor[1]])
                self.cur_cursor[1] = (self.cur_cursor[1] + 1) % dimension

                if self.cur_cursor[1] <= self.cur_cursor[0]:
                    self.cur_cursor[0] = (self.cur_cursor[0] + 1) % dimension
                    self.cur_cursor[1] = self.cur_cursor[0]

            elif self.cur_order == 3:
                term_index.append([self.cur_cursor[0], self.cur_cursor[1], self.cur_cursor[2]])
                self.cur_cursor[2] = (self.cur_cursor[2] + 1) % dimension

                if self.cur_cursor[2] <= self.cur_cursor[1]:
                    self.cur_cursor[1] = (self.cur_cursor[1] + 1) % dimension
                    self.cur_cursor[2] = self.cur_cursor[1]

                    if self.cur_cursor[1] <= self.cur_cursor[0]:
                        self.cur_cursor[0] = (self.cur_cursor[0] + 1) % dimension
                        self.cur_cursor[1] = self.cur_cursor[0]
                        self.cur_cursor[2] = self.cur_cursor[1]
            elif self.cur_order == 4:
                term_index.append([self.cur_cursor[0], self.cur_cursor[1], self.cur_cursor[2], self.cur_cursor[3]])
                self.cur_cursor[3] = (self.cur_cursor[3] + 1) % dimension

                if self.cur_cursor[3] <= self.cur_cursor[2]:
                    self.cur_cursor[2] = (self.cur_cursor[2] + 1) % dimension
                    self.cur_cursor[3] = self.cur_cursor[2]

                    if self.cur_cursor[2] <= self.cur_cursor[1]:
                        self.cur_cursor[1] = (self.cur_cursor[1] + 1) % dimension
                        self.cur_cursor[2] = self.cur_cursor[1]
                        self.cur_cursor[3] = self.cur_cursor[2]

                        if self.cur_cursor[1] <= self.cur_cursor[0]:
                            self.cur_cursor[0] = (self.cur_cursor[0] + 1) % dimension
                            self.cur_cursor[1] = self.cur_cursor[0]
                            self.cur_cursor[2] = self.cur_cursor[1]
                            self.cur_cursor[3] = self.cur_cursor[2]
            elif self.cur_order == 5:
                term_index.append([self.cur_cursor[0], self.cur_cursor[1], self.cur_cursor[2], self.cur_cursor[3],
                                   self.cur_cursor[4]])
                self.cur_cursor[4] = (self.cur_cursor[4] + 1) % dimension

                if self.cur_cursor[4] <= self.cur_cursor[3]:
                    self.cur_cursor[3] = (self.cur_cursor[3] + 1) % dimension
                    self.cur_cursor[4] = self.cur_cursor[3]

                    if self.cur_cursor[3] <= self.cur_cursor[2]:
                        self.cur_cursor[2] = (self.cur_cursor[2] + 1) % dimension
                        self.cur_cursor[3] = self.cur_cursor[2]
                        self.cur_cursor[4] = self.cur_cursor[3]

                        if self.cur_cursor[2] <= self.cur_cursor[1]:
                            self.cur_cursor[1] = (self.cur_cursor[1] + 1) % dimension
                            self.cur_cursor[2] = self.cur_cursor[1]
                            self.cur_cursor[3] = self.cur_cursor[2]
                            self.cur_cursor[4] = self.cur_cursor[3]

                            if self.cur_cursor[1] <= self.cur_cursor[0]:
                                self.cur_cursor[0] = (self.cur_cursor[0] + 1) % dimension
                                self.cur_cursor[1] = self.cur_cursor[0]
                                self.cur_cursor[2] = self.cur_cursor[1]
                                self.cur_cursor[3] = self.cur_cursor[2]
                                self.cur_cursor[4] = self.cur_cursor[3]

        return term_index
