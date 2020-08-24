import abc
import csv
import numpy as np

from typing import List, Tuple

ML_1M_DATA_FILE = r"datasets\ml-1m\ratings.dat"
ML_100K_DATA_FILE = r"datasets\ml-100k\u.data"
JESTER_100K_DATA_FILE = r"datasets\jester-100k\Jester.csv"


class UserRating:
    """
    Class representing a specific rating of user on item.
    """

    def __init__(self, user_id: int, item_id: int, rating: int = 1):
        self.user, self.item, self.rating = user_id, item_id, rating


class RatingData:
    """
    Class representing a dataset of user-item ratings.
    """

    def __init__(self, num_users=0, num_items=0):
        self.N, self.M = num_users, num_items
        self.user_ratings = []
        self.users = []
        self.items = []
        self.ratings = []
        self.matrix = None

    def __len__(self) -> int:
        return len(self.user_ratings)

    def __iter__(self) -> UserRating:
        for rating in self.user_ratings:
            yield rating

    def add(self, user: int, item: int, rating: int = 1) -> None:
        """
        Adds a new rating to the dataset.
        :param user: Zero-based, user identifier.
        :param item: Zero-based, item identifier.
        :param rating: The rating of the given user for the given item.
        """
        self.user_ratings += [UserRating(user, item, rating)]
        self.users += [user]
        self.items += [item]
        self.ratings += [rating]

    def as_matrix(self, force=False) -> np.ndarray:
        """
        Creates N by M matrix containing the rating data for cell u,i if user u had interacted with item i,
        and null otherwise. Caches matrix for future calls.
        :param force: Force matrix re-calculation
        :return: Matrix representation of the rating data.
        """
        if self.matrix is not None and not force:
            return self.matrix
        rating_matrix = np.empty((self.N, self.M))
        rating_matrix[:] = np.NaN
        for rating in self.user_ratings:
            rating_matrix[rating.user][rating.item] = rating.rating
        self.matrix = rating_matrix
        return rating_matrix

    def as_lists(self) -> Tuple[List[int], List[int], List[int]]:
        """
        :return: Triple list representation of the rating data.
        """
        return self.users, self.items, self.ratings

    def shuffle(self) -> None:
        """
        Shuffles dataset internally.
        """
        temp = list(zip(self.user_ratings, self.users, self.items, self.ratings))
        np.random.shuffle(temp)
        self.user_ratings, self.users, self.items, self.ratings = (list(x) for x in zip(*temp))

    def copy_to(self, other, start: int = None, end: int = None) -> None:
        """
        Copies all self parameters into other instance of this class.
        :param start: Add part of the dataset, starting from this index.
        :param end: Add part of the dataset, finshing on this index.
        :param other: The other instance to copy to.
        """
        other.N = self.N
        other.M = self.M
        other.user_ratings = self.user_ratings[start:end]
        other.users = self.users[start:end]
        other.items = self.items[start:end]
        other.ratings = self.ratings[start:end]

    def generate_negatives(self, sampled_negatives_per: int) -> Tuple[List[int], List[int], List[int]]:
        """
        Generates negative samples, certain amount per positive sample.
        :param sampled_negatives_per: The amount of negative samples to sample per positive sample.
        :return: A triple list representation of the negative samples (with ratings = 0)
        """
        neg_users = []
        neg_items = []
        neg_ratings = []
        temp_matrix = self.as_matrix()
        for rating in self.user_ratings:
            for _ in range(sampled_negatives_per):
                new = np.random.randint(self.M)
                while not np.isnan(temp_matrix[rating.user][new]):
                    new = np.random.randint(self.M)
                neg_users += [rating.user]
                neg_items += [new]
                neg_ratings += [0]
        return neg_users, neg_items, neg_ratings


class CFMethod(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def train(self, rating_matrix: RatingData) -> None:
        """
        Supplies the CF method with any data before making a prediction, by an input of a rating matrix
        containing values for users and items that had interacted in some way.
        :param rating_matrix: RatingData instance containing sparse representation of a rating matrix.
                              Contains value for user u and item i if user u had interacted with item i.
        """
        raise NotImplementedError("Collaborative Filtering methods must implement 'train'")

    @abc.abstractmethod
    def predict(self, user: int, item: int) -> int:
        """
        Predicts whether a user will be interested in an item.
        :param user: User to predict for
        :param item: Item to predict for
        :return: an integer representing if a user will be interested in this item (binary) or how much (scale)
        """
        raise NotImplementedError("Collaborative Filtering methods must implement 'predict'")


def load_movielens_1m(add_rating=False) -> RatingData:
    out = RatingData(6040, 3952)
    with open(ML_1M_DATA_FILE, "r") as f:
        for x, line in enumerate(f):
            u, i, r, t = (int(x) for x in line.strip().split("::"))
            if not add_rating:
                r = 1
            out.add(u - 1, i - 1, r)
    return out


def load_movielens_100k(add_rating=True) -> RatingData:
    out = RatingData(944, 1683)
    with open(ML_100K_DATA_FILE, "r") as f:
        for x, line in enumerate(f):
            u, i, r, t = (int(x) for x in line.strip().split("\t"))
            if not add_rating:
                r = 1
            else:
                r /= 5
            out.add(u - 1, i - 1, r)
    return out


def load_jester_100k(add_rating=True) -> RatingData:
    out = RatingData(7699, 159)
    with open(JESTER_100K_DATA_FILE, "r", newline='') as f:
        csv_r = csv.reader(f)
        for user, row in enumerate(csv_r):
            i_row = [float(x) for x in row]
            if i_row[0] == 1:
                continue
            for item, rating in enumerate(i_row[1:]):
                if rating != 99.:
                    if not add_rating:
                        rating = 1
                    else:
                        rating = (rating + 10) / 20
                    out.add(user, item, rating)
    return out


def _split_train_test(rating_data: RatingData, test_percent: float = 1., shuffle=True) -> Tuple[RatingData, RatingData]:
    """
    Splits a given dataset into train/test pair with given percent amount.
    :param rating_data: The whole RatingData instance to split from.
    :param test_percent: The relative percentage of the test data set.
    :param shuffle: Whether to shuffle the output datasets.
    :return: A tuple containing two RatingData instances - the first one is for training and the second for testing.
    """
    if shuffle:
        temp = RatingData()
        rating_data.copy_to(temp)
        temp.shuffle()
    else:
        temp = rating_data

    train = RatingData(temp.N, temp.M)
    temp.copy_to(train, end=int(len(temp.ratings) * (1 - (test_percent / 100))))
    test = RatingData(rating_data.N, rating_data.M)
    temp.copy_to(test, start=int(len(temp.ratings) * (1 - (test_percent / 100))))
    return train, test
