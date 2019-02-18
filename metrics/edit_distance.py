# A python implementation of Edit Distance, outputs 3 distances: substitution, deletion, insertion
import numpy as np


class DPMemorizedDistance:
    # Order for edit distance: S,D,I
    def edit_distance(self, word1, word2):
        # Wrapper for _edit_distance with empty computed_solutions
        word1 = tuple(word1)
        word2 = tuple(word2)
        return self._edit_distance(word1, word2, {})

    def _edit_distance(self, word1, word2, computed_solutions):
        # computed_solutions: dict{ tuple of word(w1,w2), tuple of integer(S,D,I)}
        if len(word1) == 0:
            return 0, 0, len(word2)  # Insert into word1
        if len(word2) == 0:
            return 0, len(word1), 0  # Delete from word1

        replace_tuple = (word1[1:], word2[1:])
        delete_tuple = (word1[1:], word2)
        insert_tuple = (word1, word2[1:])

        replace_dist = self._distance_add(self._replace_cost(word1, word2), self._transformation_cost(replace_tuple, computed_solutions))
        delete_dist = self._distance_add((0, 1, 0), self._transformation_cost(delete_tuple, computed_solutions))
        insert_dist = self._distance_add((0, 0, 1), self._transformation_cost(insert_tuple, computed_solutions))

        min_dist = self._distance_min(replace_dist, delete_dist, insert_dist)
        return min_dist

    def _replace_cost(self, word1, word2):
        """
        Cost of replacing 1st char of word1 to word2.
        :param word1:
        :param word2:
        :return: distance S,D,I
        """
        if word1[0] == word2[0]:
            return 0,0,0  # 1st chars are equal in A and B
        else:
            return 1,0,0  # Substitute 1st char in A to B

    def _transformation_cost(self, problem_tuple, solutions):
        """
        Use solution if case "tuple" was already solved, else solve the "tuple" and save to solutions
        :param problem_tuple:
        :param solutions:
        :return:
        """
        if problem_tuple in solutions:  # Already solved
            return solutions.get(problem_tuple)
        else:
            distSDI = self._edit_distance(problem_tuple[0], problem_tuple[1], solutions)  # Compute and add to solutions
            solutions[problem_tuple] = distSDI
            return distSDI

    def _distance_add(self, dis1, dis2):
        """
        Add S,D,I separately from distance
        :param dis1:
        :param dis2:
        :return:
        """

        S = dis1[0] + dis2[0]
        D = dis1[1] + dis2[1]
        I = dis1[2] + dis2[2]
        return (S,D,I)

    def _distance_min(self, dis1, dis2, dis3):
        """
        Find minimum distance from distance tuple
        :param dis1:
        :param dis2:
        :return:
        """

        d1_total = dis1[0] + dis1[1] + dis1[2]
        d2_total = dis2[0] + dis2[1] + dis2[2]
        d3_total = dis3[0] + dis3[1] + dis3[2]
        arr123 = np.array([d1_total, d2_total, d3_total], np.int32)
        argminimum = int(np.argmin(arr123))
        if argminimum == 0:
            return dis1
        elif argminimum == 1:
            return dis2
        elif argminimum == 2:
            return dis3
        else:
            raise ValueError()

def SDI(words, labels):
    """
    Compute Edit distance
    :param words: string
    :param labels: string
    :return: tuple contains (S,D,I) for times of substitute, delete, insert
    """
    dist = DPMemorizedDistance()
    dist_tuple = dist.edit_distance(words, labels)
    return dist_tuple

